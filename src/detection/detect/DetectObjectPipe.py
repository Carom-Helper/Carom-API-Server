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
from Singleton import Singleton
from pipe_cls import One2OnePipe, ResourceBag
from detect_utills import (PipeResource, LoadImages,
                           make_padding_image, copy_piperesource,
                           LOGGER, is_test, Annotator, cv2,
                           select_device, time_sync, 
                           check_img_size, non_max_suppression, xyxy2xywh, scale_boxes, print_args,
                           DetectMultiBackend)

def is_test_detect_object()->bool:
    return True and is_test()

def test_print(s, s1="", s2="", s3="", s4="", s5="", end="\n"):
    if is_test_detect_object():
        print("detect object pipe test : ", s, s1, s2, s3, s4, s5, end=end)


############################
from threading import Lock
class DetectObjectWeight(metaclass=Singleton):
    def __init__(
        self,
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=7,
        cls=[0, 1],
        imgsz=(640,640),
        device= '0'
        ) -> None:
        # 고정값
        WEIGHTS = WEIGHT_DIR / "yolo_ball.pt"
        self.yolo_weights = WEIGHTS
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
        self.model = DetectMultiBackend(
            self.yolo_weights, device=self.device, dnn=False, data=None, fp16=False)
        ############
        
        self.imgsz = check_img_size(
            self.imgsz, s=self.model.stride)  # check image size
        
        self.model.warmup(imgsz=(1, 3, *self.imgsz))  # warmup

class DetectObjectPipe(One2OnePipe):  
    def __init__(self, device, display=True):
        super().__init__()
        self.display = display
        t1 = time.time()
        
        #load model
        instance = DetectObjectWeight(device=device)
        self.model = instance.model
        self.device = instance.device
        self.conf_thres = instance.conf_thres
        self.iou_thres = instance.iou_thres
        self.max_det = instance.max_det
        self.imgsz = instance.imgsz
        self.model_instance = instance
        
        t2 = time.time()
        if display:
            print(f'[YOLOv5 init {(t2-t1):.1f}s]')

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
        device = self.device
        model = self.model
        dt, seen = [0.0, 0.0, 0.0, 0.0], 0

        conf_thres = self.conf_thres
        iou_thres = self.iou_thres
        max_det = self.max_det

        # Padding
        t1 = time_sync()
        
        im = make_padding_image(input.im) # add padding
        im = torch.from_numpy(im).to(device)
        im = im.float()  # im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        with self.model_instance.lock:
            pred = model(im, augment=False, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], input.im.shape).round()
            ## output 설정 ###
            for xmin, ymin, xmax, ymax, conf, cls in reversed(det):
                output_det = {"xmin": int(xmin), "ymin": int(ymin), "xmax": int(
                    xmax), "ymax": int(ymax), "conf": float(conf), "cls": int(cls)}
                input.dets.append(output_det)
        output = copy_piperesource(input)
        t2 = time.time()
        if self.display:
            detect_len = output.len_detkey_match("cls", "1")
            detect_len = "" if detect_len == 3 else f"(det ball :{str(detect_len)})"
            print(f'[{detect_len}YOLOv5 run {t2-t1:.3f}s]')
        
        output.print(on=(is_test_detect_object() and output.__len__() < 7))
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
    gpu = DetectObjectPipe(device="cpu")
    #cpu = DetectObjectPipe(device="0")
    
    print(id(gpu.model))
    #print(id(cpu.model))

def runner(args):
    print_args(vars(args))
    test(args.src, args.device, args.display)
    #test_singleton()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default= (CAROM_BASE_DIR / "media" / "test2"))
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--display', default=True, action="store_false")    
    args = parser.parse_args()
    runner(args) 

  
