python test_davis17_tpn_msn.py \
	--data_root=data/DAVIS-2017-trainval/ \
	--data_list=data/list/DAVIS_2017_val.txt \
	--resize_h=448 \
	--resize_w=832 \
	--gpu='0' \
	--predict_dir=save/davis17_msn \
	--restore=pretrained_models/propagation/davis17.pth \
	--restore_select=pretrained_models/selection/davis17.pth

