from __future__ import division
import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F

# general libs
from PIL import Image
import numpy as np
import tqdm
import os
import argparse

### My libs
from dataset import DAVIS_MO_Test, YTVOS_val

torch.set_grad_enabled(False) # Volatile

def get_arguments():
    parser = argparse.ArgumentParser(description="SST")
    parser.add_argument("-gpu", type=str, help="0; 0,1; 0,3; etc", default='0')
    parser.add_argument("-dataset", type=str, help="name of dataset", required=True)
    parser.add_argument("-select_file", type=str, help="path to selection file", default=None)
    parser.add_argument("-data_root", type=str, help="path to data root", default='data/DAVIS/DAVIS-2017-trainval')
    parser.add_argument("-method", type=str, help="propagate type (baseline | msn)", required=True)
    return parser.parse_args()

def Run_baseline_davis(Fs, Ms, num_frames, num_objects):
    Es = torch.zeros_like(Ms)
    Es[:, :, 0] = Ms[:, :, 0]

    for t in tqdm.tqdm(range(1, num_frames)):
        # Use previous frame result as guidance
        with torch.no_grad():
            prev_key, prev_value = model(Fs[:, :, t - 1], Es[:, :, t - 1], torch.tensor([num_objects]))

        # segment
        with torch.no_grad():
            logit = model(Fs[:, :, t], prev_key, prev_value, torch.tensor([num_objects]))
        Es[:, :, t] = F.softmax(logit, dim=1)

    pred = np.argmax(Es[0].cpu().numpy(), axis=0).astype(np.uint8)
    return pred, Es

def Run_baseline_ytvos(Fs, Ms, num_frames, num_objects, start_frame):
    # initialize start frames
    ref_frames = {}
    Es = torch.zeros_like(Ms)
    assert len(start_frame.keys()) >= 1
    for oid in start_frame.keys():
        fid = int(start_frame[oid])
        if fid not in ref_frames:
            ref_frames[fid] = []
        ref_frames[fid].append(oid)
        Es[:, oid, fid] = Ms[:, oid, fid]

    for t in tqdm.tqdm(range(1, num_frames)):
        # Use previous frame result as guidance
        with torch.no_grad():
            prev_key, prev_value = model(Fs[:, :, t - 1], Es[:, :, t - 1], torch.tensor([num_objects]))

        # segment
        with torch.no_grad():
            logit = model(Fs[:, :, t], prev_key, prev_value, torch.tensor([num_objects]))
        Es[:, :, t] = F.softmax(logit, dim=1)

        # reset logits of the annotated frame
        if t in ref_frames.keys():
            for oid in ref_frames[t]:
                Es[:, oid, t] = Ms[:, oid, t]

    pred = np.argmax(Es[0].cpu().numpy(), axis=0).astype(np.uint8)
    return pred, Es

def Run_msn_davis(Fs, Ms, num_frames, num_objects, vid_select=None, seq_name=None):
    Es = torch.zeros_like(Ms)
    Es[:, :, 0] = Ms[:, :, 0]

    select_keys = []
    select_vals = []
    select_num = 1
    for t in tqdm.tqdm(range(1, num_frames)):
        # memorize
        with torch.no_grad():
            prev_key, prev_value = model(Fs[:, :, t - 1], Es[:, :, t - 1], torch.tensor([num_objects]))
        select_keys.append(prev_key)
        select_vals.append(prev_value)

        # select keys, we only select one guidance frame here
        cur_keys = []
        cur_vals = []
        tpl = (seq_name, t)
        for k in range(select_num):
            cur_keys.append(select_keys[vid_select[tpl][k][0]])
            cur_vals.append(select_vals[vid_select[tpl][k][0]])
        cur_keys = torch.cat(cur_keys, 3)
        cur_vals = torch.cat(cur_vals, 3)

        # segment
        with torch.no_grad():
            logit = model(Fs[:, :, t], cur_keys, cur_vals, torch.tensor([num_objects]))
        Es[:, :, t] = F.softmax(logit, dim=1)

    pred = np.argmax(Es[0].cpu().numpy(), axis=0).astype(np.uint8)
    return pred, Es

def Run_msn_ytvos(Fs, Ms, num_frames, num_objects, start_frame, vid_select=None,
              seq_name=None, frame_ids=[], frameid2idx={}):
    ref_frames = {}
    Es = torch.zeros_like(Ms)
    if len(start_frame.keys()) < 1:
        assert False
    for oid in start_frame.keys():
        fid = int(start_frame[oid])
        if fid not in ref_frames:
            ref_frames[fid] = []
        ref_frames[fid].append(oid)
        Es[:, oid, fid] = Ms[:, oid, fid]
    select_num = 1
    select_keys = [] # [1, obj_ind, C, T, H, W]
    select_vals = []
    for t in tqdm.tqdm(range(1, num_frames)):
        with torch.no_grad():
            prev_key, prev_value = model(Fs[:,:,t-1], Es[:,:,t-1], torch.tensor([num_objects]))
        select_keys.append(prev_key)
        select_vals.append(prev_value)
        logit = torch.zeros_like(Es[:,:,t]).cuda().float() - 100     # [1, K, h, w]
        tmp_logit = torch.zeros((num_objects, Es.shape[3], Es.shape[4]))
        for obj_ind in range(1, num_objects+1):
            tpl = (seq_name, frame_ids[t][0], obj_ind)
            if tpl not in vid_select:
                cur_keys = select_keys[-1]
                cur_vals = select_vals[-1]
            else:
                cur_keys = select_keys[frameid2idx[vid_select[tpl][0]]]
                cur_vals = select_vals[frameid2idx[vid_select[tpl][0]]]
            with torch.no_grad():
                cur_logit = model(Fs[:,:,t], cur_keys, cur_vals, torch.tensor([num_objects]))
            tmp_logit[obj_ind - 1, :, :] = cur_logit[obj_ind - 1, :, :].clone()
        logit = model.module.Soft_aggregation(tmp_logit, Es.shape[1])
        Es[:,:,t] = F.softmax(logit, dim=1)
        if t in ref_frames.keys():
            for oid in ref_frames[t]:
                Es[:, oid, t] = Ms[:, oid, t] + 1e-3

    pred = np.argmax(Es[0].cpu().numpy(), axis=0).astype(np.uint8)
    return pred, Es


# Set up Environment
args = get_arguments()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

palette = Image.open('00000.png').getpalette()
exp_name = '%s_%s' % (args.dataset, args.method)
print('Start testing for %s' % (exp_name))

# Set up data loader
if args.dataset == 'davis16':
    Testset = DAVIS_MO_Test(args.data_root, resolution='480p', imset='2016/val.txt', single_object=True)
    Testloader = data.DataLoader(Testset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
elif args.dataset == 'davis17':
    Testset = DAVIS_MO_Test(args.data_root, resolution='480p', imset='2017/val.txt', single_object=False)
    Testloader = data.DataLoader(Testset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
elif args.dataset == 'ytvos':
    Testset = YTVOS_val(args.data_root)
    Testloader = data.DataLoader(Testset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
else:
    assert False, 'dataset %s have not been implemented' % (args.dataset)

# Set up model
if args.dataset == 'ytvos' and args.method == 'msn':
    from model_mul import STM
else:
    from model import STM
model = nn.DataParallel(STM())
if torch.cuda.is_available():
    model.cuda()
model.eval()

pth_path = 'STM_weights.pth'
print('Loading weights:', pth_path)
model.load_state_dict(torch.load(pth_path))

# Data processing
for seq, V in enumerate(Testloader):
    Fs, Ms, num_objects, info = V
    seq_name = info['name'][0]
    num_frames = info['num_frames'][0].item()
    start_frame = info.get('start_frame', None)
    frame_ids = info.get('frame_ids', None)
    if frame_ids is not None:
        frameid2idx = {}
        for idx, fid in enumerate(frame_ids):
            frameid2idx[fid[0]] = idx
    print('[{}]: num_frames: {}, num_objects: {}'.format(seq_name, num_frames, num_objects[0][0]))
    print('Start frames: ' + str(start_frame))

    if 'davis' in args.dataset:
        if args.method == 'baseline':
            pred, Es = Run_baseline_davis(Fs, Ms, num_frames, num_objects)
        elif args.method == 'msn':
            vid_select = {}
            with open(args.select_file, 'r') as fr:
                for line in fr.readlines():
                    line_s = line.strip().split(',')
                    tpl = (line_s[0], int(line_s[1]))
                    if tpl not in vid_select:
                        vid_select[tpl] = []
                    vid_select[tpl].append((int(line_s[-2]), float(line_s[-1])))
            pred, Es = Run_msn_davis(Fs, Ms, num_frames, num_objects, vid_select=vid_select, seq_name=seq_name)
        else:
            assert False, 'Not implement for type %s in dataset %s. ' % (args.method, args.dataset)
    elif 'ytvos' in args.dataset:
        if args.method == 'baseline':
            pred, Es = Run_baseline_ytvos(Fs, Ms, num_frames, num_objects, start_frame)
        elif args.method == 'msn':
            vid_select = {}
            with open(args.select_file, 'r') as fr:
                for line in fr.readlines():
                    line_s = line.strip().split(',')  # [vid_name, frame_id, obj_id, select_rank, guidance_frame_id]
                    tpl = (line_s[0], line_s[1], int(line_s[2]))
                    if tpl not in vid_select:
                        vid_select[tpl] = []
                    vid_select[tpl].append(line_s[-1])
            pred, Es = Run_msn_ytvos(Fs, Ms, num_frames, num_objects, start_frame,
                                    vid_select=vid_select, seq_name=seq_name, frame_ids=frame_ids, frameid2idx=frameid2idx)
        else:
            assert False, 'Not implement for type %s in dataset %s. ' % (args.method, args.dataset)
    else:
        assert False, 'Not implement for %s. ' % (args.dataset)

    # Save results
    test_path = os.path.join('./save', exp_name, seq_name)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    if frame_ids:
        for f in range(num_frames):
            img_E = Image.fromarray(pred[f])
            img_E.putpalette(palette)
            img_E.save(os.path.join(test_path, '%s.png' % (frame_ids[f][0])))
    else:
        for f in range(num_frames):
            img_E = Image.fromarray(pred[f])
            img_E.putpalette(palette)
            img_E.save(os.path.join(test_path, '{:05d}.png'.format(f)))


