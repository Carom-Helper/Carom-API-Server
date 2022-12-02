import torch
import numpy as np
import argparse

# set path
import os
from pathlib import Path
import sys

CAROM_BASE_DIR=Path(__file__).resolve().parent.parent.parent
FILE = Path(__file__).resolve()
ROOT = FILE.parent

tmp = ROOT
if str(tmp) not in sys.path and os.path.isabs(tmp):
    sys.path.append(str(tmp))  # add ROOT to PATH

# tmp = ROOT / "gpu_yolov5"
# if str(tmp) not in sys.path and os.path.isabs(tmp):
#     sys.path.append(str(tmp))  # add ROOT to PATH


# # ADD gpu_yolov5 to env list
# tmp = ROOT / 'gpu_yolov5'
# if str(tmp) not in sys.path and os.path.isabs(tmp):
#     sys.path.append(str(tmp))  # add yolov5 ROOT to PATH

from pipe_cls import One2OnePipe, ConvertToxywhPipe, IObserverPipe, SplitCls, ResourceOne, ResourceBag
from pipe_utills import SaveBallCoordPipe
from Singleton import Singleton
from ProjectionPipe import ProjectionCoordPipe
from DetectObjectPipe import DetectObjectPipe # ,NPU_YOLO_DIR, GPU_YOLO_DIR
from detect_utills import (PipeResource, LoadImages,
                           is_test, cv2, print_args)

def is_test_factory()->bool:
    return False and is_test()

def test_print(s, s1="", s2="", s3="", s4="", s5="", end="\n"):
    if is_test_factory():
        print("factory pipe test : ", s, s1, s2, s3, s4, s5, end=end)

class PipeFactory(metaclass=Singleton):
    def __init__(self, start_pipe=None, device='furiosa', display = True, inDB=True):
        self.pipe, _ = pipe_factory(start_pipe=start_pipe, device=device, display=display, inDB=inDB)

def pipe_factory(start_pipe=None, device='furiosa',  display=True, inDB=True):
    if display:
        print("initialize weights")
    #detect class and split class
    # projection_pipe = ProjectionPipe()
    detect_cls_pipe = DetectObjectPipe(device=device, display=display)
    xyxy2xywh = ConvertToxywhPipe()
    split_cls_pipe = SplitCls()
    edge_bag = ResourceBag()
    projection_coord_pipe = ProjectionCoordPipe(display = display)
    
    # - connect
    detect_cls_pipe.connect_pipe(xyxy2xywh)     #detect class - split_cls
    xyxy2xywh.connect_pipe(projection_coord_pipe)     #detect class - split_cls
    projection_coord_pipe.connect_pipe(split_cls_pipe)
    _ = split_cls_pipe.connect_pipe(edge_bag) # split class - edge bag

    test_print("connect edge pipe : ", _)        
    
    ball_bag = SaveBallCoordPipe(display=display) if inDB else ResourceOne()
    _ = split_cls_pipe.connect_pipe(ball_bag) # split class - ball bag
    test_print("connect ball pipe : ", _)     
    
    #set start_pipe end_pipe
    if start_pipe is None:
        start_pipe = detect_cls_pipe
    elif isinstance(start_pipe, IObserverPipe):
        start_pipe.connect_pipe(detect_cls_pipe)
    else:
        raise TypeError("TypeError in pipe_factory")
    return start_pipe, ball_bag



def detect(src, device='cpu', MIN_DETS= 10, display=False, inDB=False):
    # # set pipe
    pipe, ball_bag = pipe_factory(device=device, display=display, inDB=inDB)
    
    ### Dataloader ###
    dataset = LoadImages(src)
    ### 실행 ###
    for im0, path, s in dataset:
        #point 위치 확인
        points = [[549,109],[942,111],[1270,580],[180,565]]
        pts = np.zeros((4, 2), dtype=np.float32)
        for i in range(4):
            pts[i] = points[i]
        
        sm = pts.sum(axis=1)  # 4쌍의 좌표 각각 x+y 계산
        diff = np.diff(pts, axis=1)  # 4쌍의 좌표 각각 x-y 계산

        topLeft = pts[np.argmin(sm)]  # x+y가 가장 값이 좌상단 좌표
        bottomRight = pts[np.argmax(sm)]  # x+y가 가장 큰 값이 우하단 좌표
        topRight = pts[np.argmin(diff)]  # x-y가 가장 작은 것이 우상단 좌표
        bottomLeft = pts[np.argmax(diff)]  # x-y가 가장 큰 값이 좌하단 좌표
        test_print(f'topLeft({type(topLeft)}):{topLeft} | ({type(bottomRight)}):{bottomRight} | ({type(topRight)}):{topRight} | ({type(bottomLeft)}):{bottomLeft}')
        

        metadata = {"path": path, "carom_id":1, "TL":topLeft, "BR":bottomRight, "TR":topRight, "BL":bottomLeft}
        images = {"origin":im0}
        input = PipeResource(im=im0, metadata=metadata, images=images, s=s)
        pipe.push_src(input)
        
        input.print()
        # 원본 정사영 영역 표시
        if display:
            origin = input.get_image()
            for i in range(4):
                origin = cv2.line(origin, (pts[i][0], pts[i][1]), (pts[(i+1)%4][0], pts[(i+1)%4][1]), (0, 255, 0), 2)
            cv2.imshow("origin", origin)
            cv2.waitKey()
    return ball_bag
    
def test(
    src = CAROM_BASE_DIR / "media" / "test2" / "sample.jpg", 
    device = '0',
    display=True
    ):
    ball_bag = detect(src, device, display=display, inDB=True)
     
    title = "test"
    ball_bag.print()
    
def runner(args):
    print_args(vars(args))
    test(args.src, args.device)
    #run(args.src, args.device)
    # detect(args.src, args.device)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default=CAROM_BASE_DIR / "media" / "test2" / "sample.jpg")
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--display', action="store_true")
    args = parser.parse_args()
    runner(args) 