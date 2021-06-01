python test_davis16_tpn.py \
	--data_root=data/DAVIS-2016/ \
	--data_list=data/list/DAVIS_2016_val.txt \
	--resize_h=448 \
	--resize_w=832 \
	--gpu='0' \
	--predict_dir=save/davis16_baseline \
	--restore=pretrained_models/propagation/davis16.pth

