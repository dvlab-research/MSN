python test_ytvos_tpn.py \
	--data_root=data/YouTube-VOS/valid/ \
	--data_list=data/list/YTVOS_val.txt \
	--resize_h=320 \
	--resize_w=640 \
	--gpu='0' \
	--predict_dir=save/ytvos_baseline \
	--predict_np_dir=save/ytvos_baseline_np \
	--restore=pretrained_models/propagation/ytvos.pth

