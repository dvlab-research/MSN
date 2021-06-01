import os
import time
import logging
import numpy as np
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable

from networks.select_pos_neg_firstann_davis16 import vgg_flowc as selection
from PIL import Image
import timeit

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--data_list', type=str)
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--restore_select', type=str, default=None)
    parser.add_argument('--resize_h', type=int, default=320)
    parser.add_argument('--resize_w', type=int, default=640)
    parser.add_argument('--rgb_max', type=float, default=255.)
    parser.add_argument('--select_file', type=str, help='for selection record')

    return parser

# get logger
def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s line %(lineno)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def image_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def label_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

def calc_iou_np(pred, label, threshold=0.5, ignore_index=255):
    pred = pred.reshape(pred.shape[0], -1)
    label = label.reshape(label.shape[0], -1)
    pred = (pred > threshold).astype(np.uint8)
    mask = (label != ignore_index).astype(np.uint8)
    intsec = (label * pred) * mask
    union = (label + pred - intsec) * mask
    return np.sum(intsec), np.sum(union)

def calc_iou(pred, label):
    intsec, union = utils.calc_iou_np(pred, label)
    if union > 0:
        iou = intsec*1./union
    else:
        iou = 1.
    return iou

def image2tensor(img, label = False):
    if label:
        img = torch.from_numpy(np.array(img)).long()
    else:
        img = torch.from_numpy(np.array(img).transpose((2, 0, 1))).float()
    return img

def normalize(img, mean, std):
    if std is None:
        for t, m in zip(img, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(img, mean, std):
            t.sub_(m).div_(s)
    return img

def net_forward(model, last_image, last_prediction, img, mean, std):
    w, h = last_image.size
    w2, h2 = img.size
    assert w == w2 and h == h2, "two images do not share same height and width"

    # transform
    last_image = normalize(image2tensor(last_image, False), mean, std)
    img = normalize(image2tensor(img, False), mean, std)
    #proposal = image2tensor(proposal, True)
    last_prediction = image2tensor(last_prediction, True)

    # shape and cuda
    last_image = last_image.unsqueeze(0).cuda(async=True)
    img = img.unsqueeze(0).cuda(async=True)
    #proposal = proposal.unsqueeze(0).cuda(async=True)
    last_prediction = last_prediction.unsqueeze(0).cuda(async=True)

    # network forward
    input = torch.cat((last_image, img, last_prediction.float().unsqueeze(1)), 1)
    torch.cuda.synchronize()
    # print('tpn')
    ts = timeit.default_timer()
    pred = model.forward(Variable(input, requires_grad=True).cuda().float())
    torch.cuda.synchronize()
    # print('tpn')
    te = timeit.default_timer()
    final_pred = pred.data.cpu().numpy()[0]
    # final_pred = pred[-1].data.cpu().numpy()[0]
    return final_pred, te-ts

def colorize(gray, palette):
    # 1*3N size list palette
    color = gray.convert('P')
    temp = np.array(color)
    palette = palette.reshape(-1)
    color.putpalette(palette)
    return color

def main():
    global args, logger, writer
    args = get_parser().parse_args()
    logger = get_logger()
    logger.info(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # setting up model
    logger.info("Creating selection model...")
    selection_network = selection().cuda()
    selection_network = torch.nn.DataParallel(selection_network).cuda()
    logger.info(selection_network)

    # restore selection model
    if args.restore_select != None:
        assert os.path.isfile(args.restore_select), "no restore file found at %s" % (args.restore_select)
        logger.info("loading from %s" % (args.restore_select))
        checkpoint = torch.load(args.restore_select)
        selection_network.load_state_dict(checkpoint['state_dict'])
        logger.info("loading success")

    # dataloader setting
    mean = [0.485, 0.456, 0.406]
    mean = [item * args.rgb_max for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * args.rgb_max for item in std]
    #model.eval()

    lines = open(args.data_list, 'r').readlines()
    print("totally %d samples." % (len(lines)))

    last_prediction = None
    last_image = None
    st_time = time.time()
    cnt = 0
    obj = None
    video_id = None
    last_video_id = None
    last_obj_id = None
    mean_iou = 0.0
    valid_cnt = 0.0
    cur_idx = 0
    images = []
    images_fuse = []
    predictions = []
    features = []
    frame_ids = []
    feat0 = None
    time_tpn = []
    time_msn = []
    tot_frame = 0

    select_f = open(args.select_file, 'w')

    for line in lines:
        cnt += 1
        line = line.strip()
        img_paths = line.split(' ')
        video_id = img_paths[0].split('/')[-2]
        frame_id = img_paths[0].split('/')[-1].split('.')[0]
        tot_frame += 1
        if len(img_paths) == 2:
            obj_id = img_paths[1].split('/')[-2]
            features = []
            frame_ids = []
            last_image = image_loader(os.path.join(args.data_root, img_paths[0]))
            last_prediction = label_loader(os.path.join(args.data_root, img_paths[1]))

            # extract vgg feature
            last_image_resize = last_image.resize((args.resize_w, args.resize_h), Image.BILINEAR)
            img_tensor = normalize(image2tensor(last_image_resize, False), mean, std).unsqueeze(0).cuda()
            feat = selection_network.module.feat_extractor.forward(
                Variable(img_tensor).cuda().float()).data.cpu().numpy()
            features.append(feat)

            last_prediction_resize = last_prediction.resize((args.resize_w, args.resize_h), Image.NEAREST)
            ann_tensor = image2tensor(last_prediction_resize, True).unsqueeze(0).unsqueeze(1).cuda().float()
            mask_img_tensor = img_tensor * ann_tensor
            feat0 = selection_network.module.feat_extractor.forward(Variable(mask_img_tensor).cuda().float())

            frame_ids.append(frame_id)
            cur_idx = 0

            last_video_id = video_id
            last_obj_id = obj_id
        elif len(img_paths) == 1:
            cur_idx += 1
            img = image_loader(os.path.join(args.data_root, img_paths[0]))

            # extract feature
            img_resize = img.resize((args.resize_w, args.resize_h), Image.BILINEAR)
            img_tensor = normalize(image2tensor(img_resize, False), mean, std).unsqueeze(0).cuda()
            feat = selection_network.module.feat_extractor.forward(Variable(img_tensor).cuda().float())
            features.append(feat.data.cpu().numpy())
            frame_ids.append(frame_id)

            # '''loop
            sim_score = []
            for tt in range(cur_idx):
                feat_tt = Variable(torch.from_numpy(features[tt])).cuda().float()
                ts = timeit.default_timer()
                select_pred = selection_network.module.corr_net.forward(feat_tt, feat, feat0)
                te = timeit.default_timer()
                time_msn.append(te-ts)
                inv_score = nn.Softmax(dim=1)(select_pred)[0][0].data.cpu().numpy()
                sim_score.append(inv_score[0])
            sim_score = np.array(sim_score)

            '''write file'''
            sort_idx = np.argsort(sim_score).tolist()
            assert len(sort_idx) > 0
            for k,tmp_idx in enumerate(sort_idx):
                select_f.write(
                     "%s,%s,%s,%d,%s\n" % (video_id, frame_id, obj_id, k, frame_ids[tmp_idx]))
                if k > math.ceil(len(frame_ids)/5.)+3:
                     break
        else:
            assert False

        if cnt % 50 == 0:
            mid_time = time.time()
            run_time = mid_time - st_time
            rem_time = run_time / cnt * (len(lines) - cnt)
            rem_h = int(rem_time) // 3600
            rem_m = int(rem_time - rem_h * 3600) // 60
            rem_s = int(rem_time - rem_h * 3600 - rem_m * 60)
            logger.info("Processing %d / %d samples, remain %d h %d m %d s" % (cnt, len(lines), rem_h, rem_m, rem_s))

    select_f.close()

if __name__ == '__main__':
    main()



