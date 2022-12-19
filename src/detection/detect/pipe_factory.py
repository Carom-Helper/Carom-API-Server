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

from pipe_cls import One2OnePipe, ConvertToxywhPipe, IObserverPipe, SplitCls, ResourceOne, ResourceBag
from DjangoSaveBall import SaveBallCoordPipe
from Singleton import PIPE_Singleton
from ProjectionPipe import ProjectionCoordPipe
from CoordFilterPipe import CoordFilterPipe
from BallGeneratePipe import BallGeneratePipe
from DetectObjectPipe import DetectObjectPipe # ,NPU_YOLO_DIR, GPU_YOLO_DIR
from ImageRotationPipe import ImageRotationPipe, ResizeingPipe
from detect_utills import (PipeResource, LoadImages,
                           aline_corner_in_dict, is_test, cv2, print_args)

def is_test_factory()->bool:
    return False and is_test()

def test_print(s, s1="", s2="", s3="", s4="", s5="", end="\n"):
    if is_test_factory():
        print("factory pipe test : ", s, s1, s2, s3, s4, s5, end=end)

class PipeFactory(metaclass=PIPE_Singleton):
    def __init__(self, start_pipe=None, device='furiosa', image_size=(1080,1920), display = True, inDB=True):
        if display: print("PipeFactory Init", device)
        self.pipe, _ = pipe_factory(start_pipe=start_pipe, device=device, image_size=image_size, display=display, inDB=inDB)

def pipe_factory(start_pipe=None, device='furiosa', image_size=(1080,1920), display=True, inDB=True):
    if display:
        print("initialize weights", device)
    #detect class and split class
    # projection_pipe = ProjectionPipe()
    detect_cls_pipe = DetectObjectPipe(device=device, display=display)
    xyxy2xywh = ConvertToxywhPipe()
    split_cls_pipe = SplitCls()
    edge_bag = ResourceBag()
    projection_coord_pipe = ProjectionCoordPipe(display = display)
    coord_filter_pipe = CoordFilterPipe()
    ball_generate_pipe = BallGeneratePipe()
    
    ###################### change###############################
    rotation_pipe = ImageRotationPipe(image_size)
    resize_pipe = ResizeingPipe(image_size)
    
    pipe = rotation_pipe
    next_pipe = resize_pipe
    ############################################################
        
    # - connect
    pipe.connect_pipe(next_pipe)
    next_pipe.connect_pipe(detect_cls_pipe)
    detect_cls_pipe.connect_pipe(xyxy2xywh)     #detect class - split_cls
    xyxy2xywh.connect_pipe(projection_coord_pipe)     #detect class - split_cls
    projection_coord_pipe.connect_pipe(coord_filter_pipe)
    coord_filter_pipe.connect_pipe(ball_generate_pipe)
    ball_generate_pipe.connect_pipe(split_cls_pipe)
    _ = split_cls_pipe.connect_pipe(edge_bag) # split class - edge bag

    test_print("connect edge pipe : ", _)        
    
    ball_bag = SaveBallCoordPipe(display=display) if inDB else ResourceOne()
    _ = split_cls_pipe.connect_pipe(ball_bag) # split class - ball bag
    test_print("connect ball pipe : ", _)     
    
    #set start_pipe end_pipe
    if start_pipe is None:
        start_pipe = pipe
    elif isinstance(start_pipe, IObserverPipe):
        start_pipe.connect_pipe(detect_cls_pipe)
    else:
        raise TypeError("TypeError in pipe_factory")
    return start_pipe, ball_bag



def detect(src, device='cpu', MIN_DETS= 10, display=False, inDB=False):
    # # set pipe
    pipe, ball_bag = pipe_factory(device=device, display=display, image_size=(720, 1280),inDB=inDB)
    
    ### Dataloader ###
    dataset = LoadImages(src)
    ### 실행 ###
    for im0, path, s in dataset:
        width = im0.shape[1]
        height = im0.shape[0]
        #point 위치 확인
        # points = [[549,109],[942,111],[1270,580],[180,565]] # sample
        # points = [[549,109],[942,111],[1270,580],[180,565]]
        # points = [[256, 330],[880, 1580],[880, 330],[256, 1580]]
        
        points = [[662, 452],[662, 752], [57, 147], [57, 1057]] # CAP3825091495947943655.jpg
        sorted_point = points.copy()
        
        topLeft = sorted_point[0]
        topRight = sorted_point[1]
        bottomLeft = sorted_point[2]
        bottomRight = sorted_point[3]
        test_print(f'topLeft({type(topLeft)}):{topLeft} | ({type(bottomRight)}):{bottomRight} | ({type(topRight)}):{topRight} | ({type(bottomLeft)}):{bottomLeft}')
        

        metadata = {"path": path, "carom_id":1, "TL":topLeft, "TR":topRight, "BL":bottomLeft, "BR":bottomRight,  "WIDTH":width, "HEIGHT":height}
        images = {"origin":im0}
        input = PipeResource(im=im0, metadata=metadata, images=images, s=s)
        pipe.push_src(input)
        
        input.print()
        result = ball_bag.src
        # 원본 정사영 영역 표시
        if display:
            result.imshow_table()
            result.imshow(name="orgin", guide_line=True)
            cv2.waitKey(9000)
    return ball_bag
    
def test(
    src = CAROM_BASE_DIR / "media" / "test2" / "sample.jpg", 
    device = '0',
    display=True
    ):
    ball_bag = detect(src, device, display=display, inDB=False)
     
    title = "test"
    ball_bag.print()
    
def runner(args):
    print_args(vars(args))
    test(args.src, args.device, display=not args.no_display)
    #run(args.src, args.device)
    # detect(args.src, args.device)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default=CAROM_BASE_DIR / "media"/ "carom" / "CAP3825091495947943655.jpg") # /'test2'/'sample.jpg' )
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--no_display', default=False, action="store_true")
    args = parser.parse_args()
    runner(args) 