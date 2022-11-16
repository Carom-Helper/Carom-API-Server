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
if os.path.isabs(tmp):
    GPU_YOLO_DIR = tmp
    
tmp = ROOT / 'npu_yolov5'
if os.path.isabs(tmp):
    NPU_YOLO_DIR = tmp  # add yolov5 ROOT to PATH

YOLO_PY = Path(__file__).resolve().parent / 'gpu_yolov5'

# Set weight directory
WEIGHT_DIR = None
tmp = ROOT / 'weights'
if str(tmp) not in sys.path and os.path.isabs(tmp):
    WEIGHT_DIR= (tmp)  # add Weights ROOT to PATH




from IWeight import IWeight
from Singleton import Singleton
from detect_utills import (
    select_device, Path, check_img_size, make_padding_image
)

class GPUDetectObjectWeight(IWeight):
    def __init__(
        self,
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=20,
        label_name=[0, 1],
        imgsz=(640,640),
        device= 'furiosa'
        ) -> None:
        if str(NPU_YOLO_DIR) in sys.path:
            sys.path.remove(str(NPU_YOLO_DIR))
        if str(GPU_YOLO_DIR) not in sys.path:
            sys.path.append(str(GPU_YOLO_DIR))  # add yolov5 ROOT to PATH
        from gpu_yolov5.models.common import DetectMultiBackend
        
        t1 = time.time()
        # 고정값
        # WEIGHTS = "yolo_ball.pt"
        # self.yolo_weights = WEIGHT_DIR / WEIGHTS
        weights = WEIGHT_DIR / "yolo_ball.pt"
        self.device = select_device(model_name="YOLOv5", device=device)
        
        # 변하는 값(입력 값)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False
        self.cls = label_name
        self.imgsz = imgsz  # inference size (height, width)
        self.lock = Lock()
        
        ### load model ###
        model = DetectMultiBackend(weights, device=device, dnn=False, data=weights/'cfg.yaml', fp16=False)
        
        # model = torch.hub.load(str(YOLO_PY), "custom", path=str(self.yolo_weights) , source="local")
        # model = model.to(self.device)
        self.model = model
        ############
        self.imgsz = check_img_size(
            self.imgsz, s=32)  # check image size

        t2 = time.time()
        print( f'[GPU YOLOv5 init {(t2-t1):.1f}s]')
        if str(GPU_YOLO_DIR) in sys.path:
            sys.path.remove(str(GPU_YOLO_DIR))
        
        
    def inference(self, im, origin_size=(640,640)):
        if str(NPU_YOLO_DIR) in sys.path:
            sys.path.remove(str(NPU_YOLO_DIR))
        if str(GPU_YOLO_DIR) not in sys.path:
            sys.path.append(str(GPU_YOLO_DIR))  # add yolov5 ROOT to PATH
        #from gpu_yolov5.models.common import DetectMultiBackend
        from gpu_yolov5.utils.general import (non_max_suppression, scale_boxes)
        
        pred =  self.model(im, self.imgsz)
        
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        
        dets = list()
        for det in enumerate(pred):
            if len(det):
                _, det = det
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], origin_size).round()
                dets = det.tolist()
        if str(GPU_YOLO_DIR) in sys.path:
            sys.path.remove(str(GPU_YOLO_DIR))
        return pred
    
    
    
    def preprocess(self, im):
        img = make_padding_image(im) # add padding
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # im.half() if half else im.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        return img