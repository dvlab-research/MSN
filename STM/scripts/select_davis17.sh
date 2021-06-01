python selection.py \
	--data_root=data/DAVIS-2017-trainval/ \
	--data_list=data/list/DAVIS_2017_val.txt \
	--resize_h=448 \
	--resize_w=832 \
	--gpu='0' \
	--select_file=select_files/davis17.txt \
	--restore_select=pretrained_models/davis17.pth



