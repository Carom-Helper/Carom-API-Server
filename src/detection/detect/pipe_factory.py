import argparse
import os
from pathlib import Path
import sys

CAROM_BASE_DIR=Path(__file__).resolve().parent.parent.parent
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].__str__()
WEIGHT_DIR = None

# tmp = ROOT
# if str(tmp) not in sys.path and os.path.isabs(tmp):
#     sys.path.append(str(tmp))  # add ROOT to PATH
tmp = ROOT + '/weights'
if str(tmp) not in sys.path and os.path.isabs(tmp):
    WEIGHT_DIR= (str(tmp))  # add Weights ROOT to PATH

temp = ROOT
ROOT = ROOT + '/yolo_sort'  # yolov5 strongsort root directory

tmp = ROOT + '/yolov5'
if str(tmp) not in sys.path and os.path.isabs(tmp):
    sys.path.append(str(tmp))  # add yolov5 ROOT to PATH
    
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

ROOT=temp

from utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from utils.general import cv2
from DetectObjectPipe import DetectObjectPipe
# from DetectIndexPipe import DetectIndexPipe, ForceSetIdPipe
from pipe_cls import IObserverPipe, ConvertToxywhPipe, PipeResource, ResourceBag, SplitCls, SplitIdx, FirstCopyPipe, StartNotDetectCutterPipe, xyxy2xywh, test_print
from CheckDetectPipe import CheckDetectPipe
from FindEdgePipe import FindEdgePipe
from ProjectionPipe import ProjectionPipe, ProjectionCoordPipe

def pipe_factory(start_pipe=None, device='cpu', display = True):
    if display:
        print("initialize weights")
    #detect class and split class
    projection_pipe = ProjectionPipe()
    detect_cls_pipe = DetectObjectPipe(device=device, display=display)
    split_cls_pipe = SplitCls()
    find_edge_pipe = FindEdgePipe()
    projection_coord_pipe = ProjectionCoordPipe()
    
    # - connect
    #projection_pipe.connect_pipe(detect_cls_pipe)
    #detect_cls_pipe.connect_pipe(split_cls_pipe)
    detect_cls_pipe.connect_pipe(projection_coord_pipe)     #detect class - split_cls
    projection_coord_pipe.connect_pipe(split_cls_pipe)
    _ = split_cls_pipe.connect_pipe(find_edge_pipe) # split class - edge bag

    test_print("connect edge pipe : ", _)        
    
    split_idx_pipe = SplitIdx()
    ball_list = []
    # ball bag create and connect
    for i in range(1):
        bag = ResourceBag()
        split_cls_pipe.connect_pipe(bag)            # split idx - ball bag (iterate)
        ball_list.append(bag)
    
    #set start_pipe end_pipe
    if start_pipe is None:
        start_pipe = detect_cls_pipe
    elif isinstance(start_pipe, IObserverPipe):
        start_pipe.connect_pipe(detect_cls_pipe)
    else:
        raise TypeError("TypeError in pipe_factory")
    return (start_pipe, ball_list,  find_edge_pipe)



def detect(src, device='cpu', MIN_DETS= 10, display=True):
     ### Dataloader ###
    source = src
    imgsz = (640, 640)
    pt = True
    stride = 32
    nr_sources = 1

    im0s = cv2.imread(source)

    # dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    # vid_path, vid_writer, txt_path = [
    #     None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # # set pipe
    pipe, ball_bags, edge_bag = pipe_factory(device=device, display=display)
    
    ### 실행 ###
    # for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        # input = PipeResource(f_num=frame_idx, path=path,
        #                              im=im, im0s=im0s, vid_cap=vid_cap, s=s)
        
    input = PipeResource(path = source, im0s=im0s)
    pipe.push_src(input)
    
    ball_bag_list = []
    
    # cut index
    for i, bag in enumerate(reversed(ball_bags)):
        cnt = 0
        for resource in bag.src_list:
            # counting det num
            if resource.__len__() > 0 :
                resource.print()
                cnt += 1
                test_print(f"cnt({cnt})")
        if cnt > MIN_DETS : 
            test_print(f"{i} bag cnt({cnt})")
            ball_bag_list.append(bag)
    
    return (ball_bag_list, edge_bag)
    
def test(src, device):
    ball_bag_list, edge_bag = detect(src, device)
     
    title = "test"
    for ball_bag in ball_bag_list:
        for ball_det in ball_bag.src_list:
            ball_det.imshow(title, idx_names=["1","2","3","4","5","6"], hide_labels=False)
    print("edge : ", edge_bag.get_edge())
    
def runner(args):
    test(args.src, args.device)
    #run(args.src, args.device)
    # detect(args.src, args.device)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--src', default=CAROM_BASE_DIR / "media" / "test" / "00054.jpg")
    parser.add_argument('--src', default=CAROM_BASE_DIR / "media" / "test2" / "sample.jpg")
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    print(args.src)
    runner(args) 