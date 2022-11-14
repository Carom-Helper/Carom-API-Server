from threading import Lock
import time
import torch


# set path
import sys
from pathlib import Path
import os

CAROM_BASE_DIR=Path(__file__).resolve().parent.parent.parent
FILE = Path(__file__).resolve()
ROOT = FILE.parent

# ADD gpu_yolov5 to env list
tmp = ROOT / 'gpu_yolov5'
if str(tmp) not in sys.path and os.path.isabs(tmp):
    sys.path.append(str(tmp))  # add yolov5 ROOT to PATH

YOLO_PY = Path(__file__).resolve().parent / 'gpu_yolov5'

# Set weight directory
WEIGHT_DIR = None
tmp = ROOT / 'weights'
if str(tmp) not in sys.path and os.path.isabs(tmp):
    WEIGHT_DIR= (tmp)  # add Weights ROOT to PATH


# from gpu_yolov5.utils.general import (non_max_suppression, scale_boxes)

from IWeight import IWeight, test_print
from Singleton import Singleton
from detect_utills import (
    select_device, Path, check_img_size
)

class DetectObjectWeight(IWeight, metaclass=Singleton):
    def __init__(
        self,
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=7,
        cls=[0, 1],
        imgsz=(640,640),
        device= '0'
        ) -> None:
        t1 = time.time()
        # 고정값
        WEIGHTS = "yolo_ball.pt"
        self.yolo_weights = WEIGHT_DIR / WEIGHTS
        self.device = select_device(model_name="YOLOv5", device=device)
        
        # 변하는 값(입력 값)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        # classes = None  # filter by class: --class 0, or --class 0 2 3
        self.cls = cls
        self.imgsz = imgsz  # inference size (height, width)
        self.lock = Lock()
        
        ### load model ###
        model = torch.hub.load(str(YOLO_PY), "custom", path=str(self.yolo_weights) , source="local")
        model = model.to(self.device)
        self.model = model
        ############
        self.imgsz = check_img_size(
            self.imgsz, s=32)  # check image size

        t2 = time.time()
        print( f'[GPU YOLOv5 init {(t2-t1):.1f}s]')
    def inference(self, im, size):
        return self.model(im, size=size)