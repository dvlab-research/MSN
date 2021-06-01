import os
import time
import logging
import numpy as np
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable

from networks.vgg_flowc_sigmoid_bid import vgg_flowc
import cv2
from PIL import Image

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--data_list', type=str)
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--restore', type=str, default=None)
    parser.add_argument('--resize_h', type=int, default=320)
    parser.add_argument('--resize_w', type=int, default=640)
    parser.add_argument('--predict_np_dir', type=str, default='np_preds/', help='save temporary numpy results')
    parser.add_argument('--predict_dir', type=str, default='preds/', help='save prediction results')
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
    last_prediction = image2tensor(last_prediction, True)

    # shape and cuda
    last_image = last_image.unsqueeze(0).cuda(async=True)
    img = img.unsqueeze(0).cuda(async=True)
    last_prediction = last_prediction.unsqueeze(0).cuda(async=True)

    # network forward
    input = torch.cat((last_image, img, last_prediction.float().unsqueeze(1)), 1)
    pred = model.forward(Variable(input, requires_grad=True).cuda().float())

    final_pred = pred[-1].data.cpu().numpy()[0]
    return final_pred

def merge(src_root, dst_root, ann_root, meta_path, jpg_root):
    palette = Image.open(os.path.join('00000.png')).getpalette()
    meta_f = open(meta_path, 'r')
    meta = json.load(meta_f)

    if not os.path.exists(dst_root):
        os.mkdir(dst_root)

    for i in os.listdir(jpg_root):
        if i == '.' or i == '..':
            continue
        jpg_dir = os.path.join(jpg_root, i)
        ann_dir = os.path.join(ann_root, i)
        #split_ann_dir = os.path.join(split_ann_root, i)
        dst_dir = os.path.join(dst_root, i)
        src_dir = os.path.join(src_root, i)
        #ori_obj_lst = os.listdir(split_ann_dir)
        ori_obj_lst = list(meta['videos'][i]['objects'].keys())
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        for j in os.listdir(jpg_dir):
            if j == '.' or j == '..':
                continue
            ann_path = os.path.join(ann_dir, j[:-4] + '.png')
            dst_path = os.path.join(dst_dir, j[:-4] + '.png')

            if os.path.exists(ann_path):
                cmd = "cp %s %s" % (ann_path, dst_path)
                os.system(cmd)
        for j in os.listdir(jpg_dir):
            if j == '.' or j == '..':
                continue
            jpg_path = os.path.join(jpg_dir, j[:-4] + '.jpg')
            w, h = Image.open(jpg_path).size
            dst_path = os.path.join(dst_dir, j[:-4] + '.png')

            pred = None
            obj_lst = []
            for k in ori_obj_lst:
                src_path = os.path.join(src_dir, k, j[:-4] + '.npy')
                if os.path.exists(src_path):
                    src_img = np.load(src_path)
                    if len(src_img.shape) == 3:
                        src_img = src_img.transpose((1, 2, 0))
                    else:
                        src_img = src_img[:, :, np.newaxis]
                    obj_lst.append(k)
                    if pred is None:
                        pred = src_img
                    else:
                        pred = np.concatenate((pred, src_img), axis = 2)
            if pred is None:
                print("%s is empty" %(dst_path))
                continue
            pred_ind = np.argmax(pred, axis=2)
            pred_tmp = pred_ind.copy()
            for idx in range(len(obj_lst)):
                pred_ind[pred_tmp == idx] = obj_lst[idx]
            pred_val = np.max(pred, axis=2)
            pred_ind[pred_val < 0.5] = 0
            pred_img = Image.fromarray(pred_ind.astype(np.uint8))
            pred_img = pred_img.resize((w, h), Image.NEAREST)
            pred_img.putpalette(palette)
            pred_img.save(dst_path)

def main():
    global args, logger, writer
    args = get_parser().parse_args()
    logger = get_logger()
    logger.info(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    logger.info("Creating propagate model...")
    model = vgg_flowc(args).cuda()
    model = torch.nn.DataParallel(model).cuda()
    logger.info(model)

    # restore propagation model
    if args.restore != None:
        assert os.path.isfile(args.restore), "no restore file found at %s" % (args.restore)
        logger.info("loading from %s" % (args.restore))
        checkpoint = torch.load(args.restore)
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("loading success")

    # dataloader setting
    mean = [0.485, 0.456, 0.406]
    mean = [item * 255.0 for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * 255.0 for item in std]
    model.eval()

    lines = open(args.data_list, 'r').readlines()
    print("totally %d samples." % (len(lines)))

    last_prediction = None
    last_image = None
    st_time = time.time()
    cnt = 0
    obj = None
    video_id = None
    cur_idx = 0
    images = []
    predictions = []
    frame_ids = []

    for line in lines:
        cnt += 1
        line = line.strip()
        img_paths = line.split(' ')
        if len(img_paths) == 3:
            images = []
            predictions = []
            frame_ids = []
            last_image = image_loader(os.path.join(args.data_root, img_paths[0]))
            last_prediction = label_loader(os.path.join(args.data_root, img_paths[1]))
            w, h = last_image.size
            resize_w = args.resize_w
            resize_h = args.resize_h
            obj = img_paths[1].split('/')[-2]
            video_id = img_paths[0].split('/')[1]
            frame_id = img_paths[0].split('/')[-1][:-4]
            if w != resize_w or h != resize_h:
                last_image = last_image.resize((resize_w, resize_h), Image.BILINEAR)
                last_prediction = last_prediction.resize((resize_w, resize_h), Image.NEAREST)

            images.append(last_image)
            predictions.append(last_prediction)

            cur_idx = 0

            save_path = os.path.join(args.predict_np_dir, video_id, obj, img_paths[0].split('/')[-1][:-4] + '.npy')
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            np.save(save_path, np.array(last_prediction) * 2.0)
        elif len(img_paths) == 2:
            frame_id = img_paths[0].split('/')[-1][:-4]
            cur_idx += 1
            img = image_loader(os.path.join(args.data_root, img_paths[0]))
            w, h = img.size
            resize_w = args.resize_w
            resize_h = args.resize_h

            if w != resize_w or h != resize_h:
                img = img.resize((resize_w, resize_h), Image.BILINEAR)

            frame_ids.append(frame_id)
            pred = net_forward(model, images[cur_idx - 1], predictions[cur_idx - 1], img, mean, std)
            pred_res = np.squeeze((pred > 0.5).astype(np.uint8), 0)

            save_not = img_paths[1]
            if save_not == "1":
                save_path = os.path.join(args.predict_np_dir, video_id, obj, img_paths[0].split('/')[-1][:-4] + '.npy')
                save_dir = os.path.dirname(save_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                np.save(save_path, pred)

            images.append(img)
            predictions.append(Image.fromarray(pred_res))
        else:
            assert False
        if cnt % 10 == 0:
            mid_time = time.time()
            run_time = mid_time - st_time
            rem_time = run_time / cnt * (len(lines) - cnt)
            rem_h = int(rem_time) // 3600
            rem_m = int(rem_time - rem_h * 3600) // 60
            rem_s = int(rem_time - rem_h * 3600 - rem_m * 60)
            logger.info("Processing %d / %d samples, remain %d h %d m %d s" % (cnt, len(lines), rem_h, rem_m, rem_s))

    logger.info('Begin to merge......')

    ann_root = os.path.join(args.data_root, 'Annotations')
    meta_path = os.path.join(args.data_root, 'meta.json')
    jpg_root = os.path.join(args.data_root, 'JPEGImages')
    merge(args.predict_np_dir, dst_root=args.predict_dir, ann_root=ann_root, meta_path=meta_path, jpg_root=jpg_root)
    logger.info('Merge complete! Remove temporary files......')
    cmd = 'rm -r %s' % (args.predict_np_dir)
    print(cmd)
    #os.system(cmd)
    logger.info('Remove success......')

if __name__ == '__main__':
    main()



