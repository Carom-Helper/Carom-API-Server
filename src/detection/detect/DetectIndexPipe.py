def is_test()->bool:
    return False

def test_print(s, s1="", s2="", s3="", s4="", s5="", end="\n"):
    if is_test():
        print("detect object pipe test : ", s, s1, s2, s3, s4, s5, end=end)

import argparse
import os

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].__str__()
WEIGHT_DIR = None

temp = ROOT
tmp = ROOT + '/weights'
if str(tmp) not in sys.path and os.path.isabs(tmp):
    WEIGHT_DIR= tmp  # add Weights ROOT to PATH

ROOT = ROOT + '/yolo_sort'  # yolov5 strongsort root directory
tmp = ROOT
if str(tmp) not in sys.path and os.path.isabs(tmp):
    sys.path.append(str(tmp))  # add ROOT to PATH
    
tmp = ROOT + '/yolov5'
if str(tmp) not in sys.path and os.path.isabs(tmp):
    sys.path.append(str(tmp))  # add yolov5 ROOT to PATH
tmp = ROOT + '/strong_sort'
if str(tmp) not in sys.path and os.path.isabs(tmp):
    sys.path.append(str(tmp))  # add strong_sort ROOT to PATH
tmp = ROOT + '/strong_sort/deep/reid'
if str(tmp) not in sys.path and os.path.isabs(tmp):
    sys.path.append(str(tmp))  # add strong_sort ROOT to PATH
    
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

ROOT=temp
test_print(sys.path)



from utils.general import  cv2
from utils.augmentations import letterbox
from utils.torch_utils import select_device
from strong_sort.strong_sort import StrongSORT
from pipe_cls import One2OnePipe, PipeResource, ResourceBag, SplitIdx, xyxy2xywh
from Singleton import Singleton

class DetectIndexWeight:
    def __init__(
        self,
        device="0",
        nr_sources = 1,         #MAX_DET + 5
        ECC = False,              # activate camera motion compensation
        MC_LAMBDA = 0.8,       # matching with both appearance (1 - MC_LAMBDA) and motion cost
        EMA_ALPHA = 0.9,         # updates  appearance  state in  an exponential moving average manner
        MAX_DIST = 0.4,          # The matching threshold. Samples with larger distance are considered an invalid match
        MAX_IOU_DISTANCE = 1.5,  # Gating threshold. Associations with cost larger than this value are disregarded.
        MAX_AGE = 300,           # Maximum number of missed misses before a track is deleted
        N_INIT = 50,              # Number of frames that a track remains in initialization phase
        NN_BUDGET = 100,         # Maximum size of the appearance descriptors gallery
        MAX_DET = 7
        ) -> None:
        # initialize StrongSORT
        WEIGHTS = WEIGHT_DIR + "/osnet_x0_25_market1501.pt"
        test_print(WEIGHTS)
        
        self.ECC = ECC
        self.N_INIT = N_INIT
        strong_sort_weights = WEIGHTS
        print("Strong Sort-",end="")
        device = select_device(device)
        half = False
        
        
        # Create as many strong sort instances as there are video sources
        self.strongsort_list = []
        for i in range(nr_sources):
            self.strongsort_list.append(
                StrongSORT(
                    strong_sort_weights,
                    device,
                    half,
                    max_dist=MAX_DIST,
                    max_iou_distance=MAX_IOU_DISTANCE,
                    max_age=MAX_AGE,
                    n_init=N_INIT,
                    nn_budget=NN_BUDGET,
                    mc_lambda=MC_LAMBDA,
                    ema_alpha=EMA_ALPHA,
                )
            )
            self.strongsort_list[i].model.warmup()

#input det form : frame, x, y, w, h, cls, conf
#output det form : frame, x, y, w, h, cls, conf id
class DetectIndexPipe(One2OnePipe):
    def __init__(self, device="0", display=True , nr_sources=1) -> None:
        super().__init__()
        self.display = display
        t1 = time.time()
        
        #set weight
        instance = DetectIndexWeight(device=device)
        self.strongsort_list = instance.strongsort_list
        self.ECC = instance.ECC
        self.N_INIT = instance.N_INIT
        
        #set output, frame
        self.outputs = [None] * nr_sources
        self.curr_frames, self.prev_frames = [None] * nr_sources, [None] * nr_sources
        
        t2 = time.time()
        if display:
            print(f'[Strong Sort Pipe init {t2-t1}s]')
        
    @torch.no_grad()
    def exe(self, input: PipeResource) -> PipeResource:#output
        t1 = time.time()
        i=0 
        
        im0 = input.im0s
        self.curr_frames[i] = im0
        frame_idx = input.f_num
        dets = input.dets
        xywhs = []
        confs = []
        cls = []
        for det in dets:
            xywhs.append([det["x"], det["y"], det["w"], det["h"]])
            confs.append(det["conf"])
            cls.append(det["cls"])
        
            
        line_thickness = 2
        names=frame_idx.__str__()
        #annotator = Annotator(im0, line_width=line_thickness, example=str(names))    
        #annotator.box_label(xyxy, label, color=colors(c, True))
        
        xywhs = np.array(xywhs)
        confs = np.array(confs)#.reshape(shape=[1,-1])
        cls = np.array(cls)
        
        # xywhs[xywhs ==''] = 0.0
        # confs[confs ==''] = 0.0
        # cls[cls ==''] = 0.0
        
        xywhs = xywhs.astype(np.float64)
        confs = confs.astype(np.float64)
        cls = cls.astype(np.int64)
        
        xywhs = torch.Tensor(xywhs)
        confs = torch.Tensor(confs)
        cls = torch.Tensor(cls)
        
        if self.ECC:  # camera motion compensation
            self.strongsort_list[i].tracker.camera_update(self.prev_frames[i], self.curr_frames[i])
            
        if det is not None and len(det):
            # pass detections to strongsort
            self.outputs[i] = self.strongsort_list[i].update(xywhs, confs, cls, im0)
            #test_print(self.outputs[i], "outputs", type(self.outputs[i]))
            # draw boxes for visualization
            if len(self.outputs[i]) > 0:
                for j, (output, conf) in enumerate(zip(self.outputs[i], confs)):
                    test_print(output)
                    try:
                        bboxes = output[0:4]
                        id = int((output[4] if output[4] >= 1 or output[4] <= 5 else 6)) - 1
                        cls = output[5]
                        conf = output[6]
                        
                        xywhs = xyxy2xywh(bboxes)
                        input.update_id(key="id", value=int(id),xywh=xywhs,cls=cls, conf=conf)
                    except KeyboardInterrupt:sys.exit()
                    except:
                        input.update_id(key="id", value=int(id),xywh=xywhs,cls=cls, conf=conf)
                    test_print("output", id, end=" ")
        else:
            self.strongsort_list[i].increment_ages()
        self.prev_frames[i] = self.curr_frames[i]
        
        input.print(on=is_test())
        
        t2 = time.time()
        if self.display:
            #check det_id state
            if input.is_detkey("id"):#all setting
                print(f'[Strong Sort run {t2-t1:.3f}s]', end = " ")
                
            elif input.len_unset_detkey("id") < 3:
                print(f'[(unset id:{input.len_unset_detkey("id")})Strong Sort run {t2-t1:.3f}s]', end = " ")
            # else:
            #     print(f'[(unset id:{input.len_unset_detkey("id")})Strong Sort run {t2-t1:.3f}s]', end = " ")
                
        return input
        
    def get_regist_type(self, idx=0) -> str:
        return "StrongSORT"

class ForceSetIdPipe(One2OnePipe):
    
    ball_num = 3
    
    def __init__(self, display=True) -> None:
        super().__init__()
        self.display = display
        self.sum_ball_idx = 0
        test_print("balls num : ", self.sum_ball_idx)
    
    def exe(self, input: PipeResource) -> PipeResource:
        ID_KEY = "id"
        CLS_KEY = "cls"
        BALL_NUM = self.ball_num
        
        dets = input.dets
        if input.len_detkey_match(CLS_KEY, 1) == BALL_NUM:# detect ball == 3
            # 첫프레임 일때 idx 초기화
            if self.sum_ball_idx == 0 :
                # ball == 3 이고 id도 3개가 있을 때 세팅
                if input.is_detkey(ID_KEY) and input.len_unset_detkey(ID_KEY) == 0:
                    for det in input.dets:
                        self.sum_ball_idx += ( int(det[ID_KEY]) )
                else :
                    input = None
            # idx_values 세팅 되어 있을 때
            if self.sum_ball_idx > 0:
                #id를 하나만 못 찾았을 때
                if input.is_detkey(ID_KEY) == False and input.len_unset_detkey(ID_KEY) == 1:
                    target_idx = -1
                    sum = self.sum_ball_idx
                    for i, det in enumerate(input.dets):
                        try:
                            sum -= int(det[ID_KEY])
                        except KeyboardInterrupt: sys.exit()
                        except:
                            target_idx = i
                    try:
                        input.dets[target_idx][ID_KEY] = sum
                        if self.display:
                            print(f'[Force setting ID run (set:{sum})]', end = " ")
                    except KeyboardInterrupt: sys.exit()
                    except Exception as ex:
                        print(self.__str__(), " set dict[id_key] : ", ex.__str__())
                    input.print(on=is_test())
                else : # +++++++++++++++++++++++++++++++error
                    pass
                
        return input
    
    def get_regist_type(self, idx=0) -> str:
        return "ForceSet_ID"
        
    
class yolofile_reader:
    # 폴더에서 같은 이름의 파일과 이미지를 매칭해서 읽는다.
    # 그 후 get_frame을 호출하면 하나씩 PipeResource를 꺼내준다.
    def __init__(self, dir_name) -> None:
        self.dir = dir_name
        dirlist = os.listdir(dir_name)
        self.namelist = [_.split(".")[0] for _ in dirlist if _.endswith(".jpg")]
        self.txtlist = [dir_name+"/"+_ for _ in dirlist if _.endswith(".txt")]
        self.imglist = [dir_name+"/"+_ for _ in dirlist if _.endswith(".jpg")]
        
        #test_print("====yolofile_reader namelist====")
        #test_print(self.namelist.__str__())
        #test_print("====yolofile_reader txtlist====")
        #test_print(self.txtlist.__str__())
        #test_print("====yolofile_reader imglist====")
        #test_print(self.imglist.__str__())
    
    def __iter__(self):
        self.count = 0
        return self

    def __next__(self) -> PipeResource:
        if self.count == self.__len__():
            raise StopIteration
        path = self.dir
        
        cnt = self.count 
        src = self.convert_file2resource(self.txtlist[cnt], self.imglist[cnt])
        self.count += 1
        return src
        
    def __len__(self):
        return self.txtlist.__len__()  # number of files
    
    def convert_file2resource(self, det_name, img_name, vid=0) ->PipeResource:
        #read img
        path=img_name
        img_size = 640
        stride = 32
        auto = True
        try:
            # Read image
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.__len__()} {path}: '

            # Padded resize
            img = letterbox(img0, img_size, stride=stride, auto=auto)[0]
        except KeyboardInterrupt:sys.exit()
        except Exception as ex:
            print(f"convert_file2resource({img_name}) : "+ ex.__str__())
        
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        
        src = PipeResource(vid=vid, f_num=self.count, path=path, im=img, im0s=img0, s=s)
        try:
            frame_num = int(img_name.split("/")[-1].split(".")[0])
        except KeyboardInterrupt:sys.exit()
        except:
            print("file name setting error")
        
        # read txt
        dets = []
        try:
            with open(det_name, 'r') as f:
                for line in f:
                    words = line.strip().split(' ')
                    #print(words)
                    det = dict()
                    i = 0
                    det["frame"] = int(words[i])#frame_num#
                    i += 1
                    det["cls"] = int(words[i])
                    i += 1
                    det["x"] = float(words[i])
                    i += 1
                    det["y"] = float(words[i])
                    i += 1
                    det["w"] = float(words[i])
                    i += 1
                    det["h"] = float(words[i])
                    i += 1
                    det["conf"] = float(words[i])
                    i += 1
                    dets.append(det)
        except KeyboardInterrupt:sys.exit()
        except Exception as ex:
            print(f"convert_file2resource({det_name}) : "+ ex.__str__())
        src.dets = dets
        
        return src

def test_yolofile_reader(src):
    file = yolofile_reader(src)
    test_print("frame length : " + file.__len__().__str__())
    
    for i, src in enumerate(file):
        src.print()
        src.imshow()
        cv2.waitKey(0)

def test_track(src):
    file = yolofile_reader(src)
    sort_pipe = DetectIndexPipe()
    
    for i, src in enumerate(file):
        output = sort_pipe.exe(src)
        if output is not None:
            #output.print()
            output.imshow("helle", hide_labels=False)
            cv2.waitKey(0)
            
def detect_index_pipe(start_pipe=None, device='cpu'):
    # create pipee
    sort_pipe = DetectIndexPipe(device=device)
    split_pipe = SplitIdx()
    
    #connect pipe
    sort_pipe.connect_pipe(split_pipe)
    bag_list = []
    print(split_pipe.idx2label)
    for i in range(split_pipe.idx2label.__len__()):
        bag = ResourceBag()
        split_pipe.connect_pipe(bag)
        bag_list.append(bag)
    
    return (sort_pipe, bag_list)


def test_push_src(src, device):
    file = yolofile_reader(src)
    # create pipe
    sort_pipe, bag_list = detect_index_pipe(src, device)
    
    #push_src
    print("push_src")
    for i, src in enumerate(file):
        sort_pipe.push_src(src)
    
    MIN_DETS= 10
    # cut index
    ball_bag_list = []
    for i, bag in enumerate(bag_list):
        cnt = 0
        print(i)
        for resource in bag.src_list:
            # counting det num
            if resource.__len__() > 0 :
                #resource.print()
                cnt += 1
                print(f"cnt({cnt})")
        if cnt > MIN_DETS : 
            test_print(f"{i} bag cnt({cnt})")
            ball_bag_list.append(bag)    
        
    
    # check result
    print("check result")
    for i, bag in enumerate(ball_bag_list):
        title = f"id:{i}"
        print(title)
        # bag.print()
        for resource in bag.src_list:
            resource.imshow(title, idx_names=["1","2","3","4","5","6"], hide_labels=False)
            cv2.waitKey(0)
    
def test_singleton():
    gpu = DetectIndexPipe(device="cpu")
    cpu = DetectIndexPipe(device="0")
    
    print(gpu.strongsort_list)
    print(cpu.strongsort_list)

def runner(args):
    #test_yolofile_reader(args.src)
    #test_track(args.src)
    #test_push_src(args.src, args.device)
    test_singleton()

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default=ROOT+"/../../data/videos/q_shot3")
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    print(args.src)
    runner(args) 