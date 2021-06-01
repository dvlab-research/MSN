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
    parser.add_argument('--predict_dir', type=str, default='preds/', help='save prediction results')
    return parser

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

def main():
    global args, logger, writer
    args = get_parser().parse_args()
    logger = get_logger()
    logger.info(args)

    palette = Image.open(os.path.join('00000.png')).getpalette()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # setting up model
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

    if not os.path.exists(args.predict_dir):
        os.mkdir(args.predict_dir)

    # dataloader setting
    mean = [0.485, 0.456, 0.406]
    mean = [item * 255.0 for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * 255.0 for item in std]

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
    predictions = []
    frame_ids = []
    predictions_objs = {}

    for line in lines:
        cnt += 1
        line = line.strip()
        img_paths = line.split(' ')
        video_id = img_paths[0].split('/')[-2]
        obj_id = img_paths[1].split('/')[-2]
        frame_id = img_paths[0].split('/')[-1].split('.')[0]

        if last_video_id is None or last_video_id != video_id or last_obj_id != obj_id:
            if last_video_id is not None and last_video_id != video_id:
                for fis in sorted(predictions_objs.keys()):
                    pred = None
                    for ois in sorted(predictions_objs[fis].keys()):
                        if pred is None:
                            pred = predictions_objs[fis][ois][0][np.newaxis, :]
                        else:
                            pred = np.concatenate((pred, predictions_objs[fis][ois][0][np.newaxis]), axis=0)
                    pred_ind = np.argmax(pred, axis=0)
                    pred_tmp = pred_ind.copy()
                    obj_list = sorted(predictions_objs[fis].keys())
                    for idx in range(len(predictions_objs[fis].keys())):
                        pred_ind[pred_tmp == idx] = obj_list[idx]
                    pred_max = np.max(pred, axis=0)
                    pred_ind[pred_max < 0.5] = 0

                    save_path = os.path.join(args.predict_dir, last_video_id, "%s.png" % (fis))
                    save_dir = os.path.dirname(save_path)
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    img_E = Image.fromarray(pred_ind.astype(np.uint8))
                    img_E.putpalette(palette)
                    img_E.save(save_path)
                predictions_objs = {}

            # set up predict and obj dict
            if frame_id not in predictions_objs.keys():
                predictions_objs[frame_id] = {}
            predictions_objs[frame_id][obj_id] = []

            images = []
            predictions = []
            frame_ids = []
            last_image = image_loader(os.path.join(args.data_root, img_paths[0]))
            last_prediction_ori = label_loader(os.path.join(args.data_root, img_paths[1]))
            last_prediction = last_prediction_ori
            w, h = last_image.size
            resize_w = args.resize_w
            resize_h = args.resize_h
            if w != resize_w or h != resize_h:
                last_image = last_image.resize((resize_w, resize_h), Image.BILINEAR)
                last_prediction = last_prediction.resize((resize_w, resize_h), Image.NEAREST)
            images.append(last_image)
            predictions.append(last_prediction)

            predictions_objs[frame_id][obj_id].append(np.array(last_prediction_ori))
            frame_ids.append(frame_id)
            cur_idx = 0

            last_video_id = video_id
            last_obj_id = obj_id
        else:
            cur_idx += 1
            img = image_loader(os.path.join(args.data_root, img_paths[0]))
            if frame_id not in predictions_objs.keys():
                predictions_objs[frame_id] = {}
            predictions_objs[frame_id][obj_id] = []
            w, h = img.size
            resize_w = args.resize_w
            resize_h = args.resize_h
            if w != resize_w or h != resize_h:
                img = img.resize((resize_w, resize_h), Image.BILINEAR)

            pred = net_forward(model, images[cur_idx - 1], predictions[cur_idx - 1], img, mean, std)
            if w != resize_w or h != resize_h:
                pred_eva = cv2.resize(pred[0], (w, h), interpolation=cv2.INTER_LINEAR)
            pred_res = np.squeeze((pred > 0.5).astype(np.uint8), 0)

            predictions_objs[frame_id][obj_id].append(pred_eva)
            images.append(img)
            predictions.append(Image.fromarray(pred_res))
            last_video_id = video_id
            last_obj_id = obj_id
        if cnt % 10 == 0:
            mid_time = time.time()
            run_time = mid_time - st_time
            rem_time = run_time / cnt * (len(lines) - cnt)
            rem_h = int(rem_time) // 3600
            rem_m = int(rem_time - rem_h * 3600) // 60
            rem_s = int(rem_time - rem_h * 3600 - rem_m * 60)
            logger.info("Processing %d / %d samples, remain %d h %d m %d s" % (cnt, len(lines), rem_h, rem_m, rem_s))

    for fis in sorted(predictions_objs.keys()):
        pred = None
        for ois in sorted(predictions_objs[fis].keys()):
            if pred is None:
                pred = predictions_objs[fis][ois][0][np.newaxis, :]
            else:
                pred = np.concatenate((pred, predictions_objs[fis][ois][0][np.newaxis]), axis=0)
        pred_ind = np.argmax(pred, axis=0)
        pred_tmp = pred_ind.copy()
        obj_list = sorted(predictions_objs[fis].keys())
        for idx in range(len(predictions_objs[fis].keys())):
            pred_ind[pred_tmp == idx] = obj_list[idx]
        pred_max = np.max(pred, axis=0)
        pred_ind[pred_max < 0.5] = 0

        save_path = os.path.join(args.predict_dir, last_video_id, "%s.png" % (fis))
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        img_E = Image.fromarray(pred_ind.astype(np.uint8))
        img_E.putpalette(palette)
        img_E.save(save_path)


if __name__ == '__main__':
    main()



