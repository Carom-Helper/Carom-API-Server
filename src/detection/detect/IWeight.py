from threading import Lock
from abc import *
import time
import torch


# set path
import sys
from pathlib import Path
import os


FILE = Path(__file__).resolve()
ROOT = FILE.parent

from detect_utills import (
    select_device, Path, is_test
)

def is_test_Weight()->bool:
    return True and is_test()

def test_print(s, s1="", s2="", s3="", s4="", s5="", end="\n"):
    if is_test_Weight():
        print("weight pipe test : ", s, s1, s2, s3, s4, s5, end=end)

class IWeight:
    cls=None # list!!!
    WEIGHTS=None # Path
    NAME=None
    ATTRBUTES_TYPE_ID=None
    def __init__(
        self,
        display=True,
        conf_thres=0.25,
        imgsz=(224, 224),
        device = '0',
        cls=None,
        weights_path=None,
        name=None,
        id=None
        ) -> None:
        if cls is not None: self.cls=cls
        if weights_path is not None: self.WEIGHTS=weights_path
        if name is not None: self.NAME=name
        if id is not None: self.ATTRBUTES_TYPE_ID=id
        t1 = time.time()
        # Select device
        self.device = select_device(model_name=self.NAME,device=device)
        # 변하는 값(입력 값)
        self.conf_thres = conf_thres
        self.imgsz = imgsz  # inference size (height, width)
        self.lock = Lock()
        self.display = display

        # weight2gpu
        model = torch.load(self.WEIGHTS)
        # if is_test_Weight():
        #     print(model)
        try:             model.eval()
        except: pass
        model = model.to(self.device)
        self.model = model
        t2 = time.time()
        print( f'[{self.NAME} init {(t2-t1):.1f}s]')
    @abstractmethod
    def inference(self, im, size):
        pass
    # @abstractmethod
    # def non_max_suppression((
    #     prediction,
    #     conf_thres=0.25,
    #     iou_thres=0.45,
    #     classes=None,
    #     agnostic=False,
    #     multi_label=False,
    #     labels=(),
    #     max_det=300,
    #     nm=0,  # number of masks
    #     ):
    #     pass