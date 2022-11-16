import torch
import time
import argparse

# set path
import sys
from pathlib import Path
import os


CAROM_BASE_DIR=Path(__file__).resolve().parent.parent.parent
FILE = Path(__file__).resolve()
ROOT = FILE.parent


# ADD gpu_yolov5 to env list
# tmp = ROOT / 'npu_yolov5'
# if str(tmp) not in sys.path and os.path.isabs(tmp):
#     sys.path.append(str(tmp))  # add yolov5 ROOT to PATH

# # ADD gpu_yolov5 to env list
# tmp = ROOT / 'gpu_yolov5'
# if str(tmp) not in sys.path and os.path.isabs(tmp):
#     sys.path.append(str(tmp))  # add yolov5 ROOT to PATH

# Set weight directory
WEIGHT_DIR = None
tmp = ROOT / 'weights'
if str(tmp) not in sys.path and os.path.isabs(tmp):
    WEIGHT_DIR= (tmp)  # add Weights ROOT to PATH



# import my project
# from Singleton import Singleton
from pipe_cls import One2OnePipe, ResourceBag
from IWeight import IWeight
from npu_weight import NPUDetectObjectWeight
from gpu_weight import GPUDetectObjectWeight
from detect_utills import (PipeResource, LoadImages, copy_piperesource,
                           is_test, cv2, print_args)

def is_test_detect_object()->bool:
    return True and is_test()

def test_print(s, s1="", s2="", s3="", s4="", s5="", end="\n"):
    if is_test_detect_object():
        print("detect object pipe test : ", s, s1, s2, s3, s4, s5, end=end)


############################
from threading import Lock

class DetectObjectPipe(One2OnePipe):
    cls_list = ["EDGE", "BALL"]
    def __init__(self, device, framework="furiosa", display=True):
        super().__init__()
        self.display = display
        t1 = time.time()
        
        #load model
        instance = NPUDetectObjectWeight(device=device) if device=="furiosa" or device=='onnx' else GPUDetectObjectWeight(device=device)
        self.model = instance
        self.lock = instance.lock
        self.framework = device
        
        t2 = time.time()
        if display:
            print(f'[{str(framework).upper()} YOLOv5 init {(t2-t1):.1f}s]')

    @torch.no_grad()
    def exe(
        self,
        input: PipeResource,
        visualize = False,
        agnostic_nms = False,
        classes = None
        ) -> PipeResource:
        t1 = time.time()
        output = PipeResource()

        # 고정 값
        model = self.model
        dt = [0.0, 0.0, 0.0, 0.0]
        
        # preprocess
        t1 = time.time()
        im = model.preprocess(input.im)
        t2 = time.time()
        dt[0] += t2 - t1
        
        # Inference
        with self.lock:
            pred = model.inference(im, input.im.shape)
        t3 = time.time()
        dt[1] += t3 - t2
        
        # Process detections
        for det in pred:# detection per image
            test_print(det)
            for xmin, ymin, xmax, ymax, conf, cls in det: # detect datas
                output_det = {"xmin": int(xmin), "ymin": int(ymin), "xmax": int(
                    xmax), "ymax": int(ymax), "conf": float(conf), "cls": int(cls), "label":self.cls_list[int(cls)]}
                input.dets.append(output_det)
        output = copy_piperesource(input)
        
        t2 = time.time()
        if self.display:
            detect_len = output.len_detkey_match("cls", "1")
            detect_len = "" if detect_len == 3 else f"(det ball :{str(detect_len)})"
            print(f'[{detect_len}YOLOv5 run {t2-t1:.3f}s {str(self.framework).upper()}]')
        
        output.print(on=(is_test_detect_object()))
        if is_test_detect_object():
            print(f'[{str(self.framework).upper()} YOLOv5 run {t2-t1:.3f}s]')
        return output

    def get_regist_type(self, idx=0) -> str:
        return "det_obj"

def test(src, device, display=True):
    ### Pipe 생성###
    detectObjectPipe1 = DetectObjectPipe(device=device, display=display)
    bag_split = ResourceBag()
    
    # 파이프 연결
    detectObjectPipe1.connect_pipe(bag_split)
    ### Dataloader ###
    dataset = LoadImages(src)
    ### 실행 ###
    for im0, path, s in dataset:
        metadata = {"path": path}
        images = {"origin":im0}
        input = PipeResource(im=im0, metadata=metadata, images=images, s=s)
        detectObjectPipe1.push_src(input)
    
    if display:
        for src in bag_split.src_list:
            src.imshow(name="hellow")
            cv2.waitKey(1000)
    else:
        bag_split.print()
        
        
def test_singleton():
    npu = DetectObjectPipe(device="npu")
    #cpu = DetectObjectPipe(device="0")
    
    print(id(npu.model))
    #print(id(cpu.model))

def runner(args):
    print_args(vars(args))
    test(args.src, args.device, not args.no_display)
    #test_singleton()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default= (CAROM_BASE_DIR / "media" / "test2"))
    parser.add_argument('--device', default='furiosa', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--no_display', default=True, action="store_false")    
    args = parser.parse_args()
    runner(args) 

  
