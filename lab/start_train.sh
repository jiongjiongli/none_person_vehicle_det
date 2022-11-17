# bash /project/train/src_repo/mmdetection/start_train.sh
cd /project/train/src_repo/mmdetection
python data_parser.py

python tools/train.py configs/cascade_mask_rcnn_non_pserson_vehicle_coco.py --gpu-id 3
