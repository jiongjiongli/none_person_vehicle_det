cd /project/train/src_repo

python -m venv py_yolov5

source /project/train/src_repo/py_yolov5/bin/activate

git clone https://github.com/ultralytics/yolov5  # clone

cd /project/train/src_repo/yolov5/
pip install -r requirements.txt  # install

import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

# Images
img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()


https://api.github.com/repos/ultralytics/yolov5/releases/tags/v7.0

yolov5s.pt

https://api.github.com/repos/ultralytics/yolov5/releases/tags/v7.0
https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt

from hubconf import _create
from pathlib import Path

# Model
model = _create(name='yolov5s', pretrained=True, channels=3, classes=80, autoshape=True, verbose=True)
# model = custom(path='path/to/model.pt')  # custom

# Images
imgs = [
    'data/images/zidane.jpg',  # filename
    # Path('data/images/zidane.jpg'),  # Path
    # 'https://ultralytics.com/images/zidane.jpg',  # URI
    # cv2.imread('data/images/bus.jpg')[:, :, ::-1],  # OpenCV
    # Image.open('data/images/bus.jpg'),  # PIL
    # np.zeros((320, 640, 3))
    ]  # numpy

# Inference
results = model(imgs, size=320)  # batched inference

# Results
results.print()
results.save()

https://ultralytics.com/assets/Arial.ttf
