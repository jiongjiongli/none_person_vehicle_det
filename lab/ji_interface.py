import argparse
import numpy as np
import cv2
import json


def init():
    """
        Initialize model
        Returns: model
    """
    return {}


def process_image(net, input_image, args=None):
    """
        Do inference to analysis input_image and get output
        Attributes:
        handle: algorithm handle returned by init()
        input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR
        args:
            模型榜时, 此参数为空。
        Returns: process result
    """
    fake_result = {}
    fake_result["algorithm_data"] = {
        "is_alert": False,
        "target_count": 0,
        "target_info": []
    }
    fake_result["model_data"] = {}
    fake_result["model_data"]["objects"] = [
        {
            "x": 1622,
            "y": 677,
            "width": 57,
            "height": 69,
            "confidence": 0.800463,
            "name": "bicycle_person"
        }
    ]
    return json.dumps(fake_result, indent=4)
