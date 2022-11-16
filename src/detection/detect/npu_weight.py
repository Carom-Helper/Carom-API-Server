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


tmp = ROOT / 'npu_yolov5'
if os.path.isabs(tmp):
    NPU_YOLO_DIR = tmp  # add yolov5 ROOT to PATH

tmp = ROOT / 'gpu_yolov5'
if os.path.isabs(tmp):
    GPU_YOLO_DIR = tmp  # add yolov5 ROOT to PATH

# Set weight directory
WEIGHT_DIR = None
tmp = ROOT / 'weights'
if str(tmp) not in sys.path and os.path.isabs(tmp):
    WEIGHT_DIR= (tmp)  # add Weights ROOT to PATH

from IWeight import IWeight
from Singleton import Singleton
from detect_utills import (
    select_device, Path
)



class NPUDetectObjectWeight(IWeight):
    def __init__(
        self,
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=7,
        cls=[0, 1],
        imgsz=(640,640),
        device= 'furiosa'
        ) -> None:
        # ADD gpu_yolov5 to env list
        if str(GPU_YOLO_DIR) in sys.path:
            sys.path.remove(str(GPU_YOLO_DIR))
        if str(NPU_YOLO_DIR) not in sys.path:
            sys.path.append(str(NPU_YOLO_DIR))  # add yolov5 ROOT to PATH
        from npu_yolov5.models.yolov5 import Yolov5Detector
        t1 = time.time()
        # 고정값
        WEIGHTS = "npu_yolo_ball"
        self.yolo_weights = WEIGHT_DIR / WEIGHTS
        self.device = select_device(model_name="FURIOSA YOLOv5", device=device)
        
        # 변하는 값(입력 값)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        # classes = None  # filter by class: --class 0, or --class 0 2 3
        self.cls = cls
        self.imgsz = imgsz  # inference size (height, width)
        self.lock = Lock()
        
        framework = device
        weights = self.yolo_weights / "weights.onnx"
        cfg_file = self.yolo_weights / "cfg.yaml"
        calib_data = CAROM_BASE_DIR / 'media' / 'test'
        calib_data_count = 10
        
        ### load model ### Yolov5Detector가 들어가야함
        model = Yolov5Detector(weights, cfg_file, framework, calib_data, calib_data_count)
        self.model = model
        ############
        
        t2 = time.time()
        print( f'[NPU YOLOv5 init {(t2-t1):.1f}s]')
        if str(NPU_YOLO_DIR) in sys.path:
            sys.path.remove(str(NPU_YOLO_DIR))
        
        
    def inference(self, im, origin_size=(640,640)):
        if str(NPU_YOLO_DIR) not in sys.path:
            sys.path.append(str(NPU_YOLO_DIR))  # add yolov5 ROOT to PATH
        from npu_yolov5.models.yolov5 import Yolov5Detector
        result = [self.model(im)]
        if str(NPU_YOLO_DIR) not in sys.path:
            sys.path.remove(str(NPU_YOLO_DIR))
        return result
    
    def preprocess(self, im):
        return im