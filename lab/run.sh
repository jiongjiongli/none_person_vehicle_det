pip install --upgrade pip

pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10/index.html
pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10/index.html
pip install mmdet==2.25.3

# git clone https://github.com/open-mmlab/mmdetection.git
git clone https://gitee.com/mirrors/mmdetection.git
cd mmdetection
wget http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

# Test mmdetection
from mmdet.apis import init_detector, inference_detector

config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
# 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'
# 初始化检测器
model = init_detector(config_file, checkpoint_file, device=device)
# 推理演示图像
inference_detector(model, 'demo/demo.jpg')