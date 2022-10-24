import argparse
import os
from pathlib import Path
import sys

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
from DetectObjectPipe import DetectObjectPipe
from DetectIndexPipe import DetectIndexPipe, ForceSetIdPipe
from pipe_cls import IObserverPipe, ConvertToxywhPipe, PipeResource, ResourceBag, SplitCls, SplitIdx, FirstCopyPipe, StartNotDetectCutterPipe, xyxy2xywh, test_print
from CheckDetectPipe import CheckDetectPipe
from FindEdgePipe import FindEdgePipe

def pipe_factory(start_pipe=None, device='cpu', display = True):
    if display:
        print("initialize weights")
    #detect class and split class
    detect_cls_pipe = DetectObjectPipe(device=device, display=display)
    split_cls_pipe = SplitCls()
    find_edge_pipe = FindEdgePipe()
    
    # - connect
    detect_cls_pipe.connect_pipe(split_cls_pipe)     #detect class - split_cls
    _ = split_cls_pipe.connect_pipe(find_edge_pipe) # split class - edge bag
    test_print("connect edge pipe : ", _)        
    
    
    #detect index and split index
    detect_idx_pipe = DetectIndexPipe(device=device, display=display)
    xywh_pipe = ConvertToxywhPipe()
    repeat_pipe = FirstCopyPipe(N_INIT=detect_idx_pipe.N_INIT, display=display)
    force_setid_pipe = ForceSetIdPipe(display=display)
    check_detect_pipe = CheckDetectPipe()
    start_cutter_pipe = StartNotDetectCutterPipe()
    
    
    #connect
    _ = split_cls_pipe.connect_pipe(start_cutter_pipe)      # split cls - ball - start cut
    test_print("connect sort pipe : ", _)
    start_cutter_pipe.connect_pipe(check_detect_pipe)       # start cut - check -detect
    check_detect_pipe.connect_pipe(xywh_pipe)               # check detect - xyxy
    xywh_pipe.connect_pipe(repeat_pipe)                     # check detect - repeat
    repeat_pipe.connect_pipe(detect_idx_pipe)               # repeat - detect idx
    detect_idx_pipe.connect_pipe(force_setid_pipe)          # detect idx - force set id
    
    split_idx_pipe = SplitIdx()
    ball_list = []
    # ball bag create and connect
    for i in range(split_idx_pipe.idx2label.__len__()):
        bag = ResourceBag()
        split_idx_pipe.connect_pipe(bag)            # split idx - ball bag (iterate)
        ball_list.append(bag)
    force_setid_pipe.connect_pipe(split_idx_pipe)    # force set id - split idx
    
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

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    vid_path, vid_writer, txt_path = [
        None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # set pipe
    pipe, ball_bags, edge_bag = pipe_factory(device=device, display=display)
    
    ### 실행 ###
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        input = PipeResource(f_num=frame_idx, path=path,
                             im=im, im0s=im0s, vid_cap=vid_cap, s=s)
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
            cv2.waitKey(0)
    print("edge : ", edge_bag.get_edge())
    
def runner(args):
    test(args.src, args.device)
    #run(args.src, args.device)
    detect(args.src, args.device)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default="/../../data/videos/kj_cud_272.mp4")
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    print(args.src)
    runner(args) 