from abc import *
from pickle import NONE
from random import random, randrange
import time

def is_test()->bool:
    return False

def test_print(s, s1="", s2="", s3="", s4="", s5="", end="\n"):
    if is_test():
        print("pipe cls exe : ", s, s1, s2, s3, s4, s5, end=end)

import sys
from pathlib import Path
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].__str__()

temp = ROOT

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


##############################################no test#####################################################
# from yolo_sort.yolov5.utils.metrics import bbox_iou
from yolo_sort.yolov5.utils.plots import Annotator, colors, save_one_box
from yolo_sort.yolov5.utils.general import (xywh2xyxy, cv2)
from DetectError import *
class PipeResource:
    def __init__(self, vid=-1, f_num=0, path=None, im=None, im0s=None, vid_cap=None, s=None) -> None:
        self.vid = vid #video id / set undefind
        self.f_num = f_num #video frame number / set undefind
        # get dateset  (path, im, im0s, vid_cap, s)
        self.path = path        # video path
        self.vid_cap = vid_cap  # video data
        self.s = s              # string
        
        self.im = im            # pading image
        
        self.im0s = im0s        # origin image
        self.dets = list()
        self.unset_det_num = dict()
    
    def __iter__(self):
        self.count = 0
        return self
    
    def __next__(self) -> dict:
        if self.count == self.__len__():
            raise StopIteration
        
        cnt = self.count 
        
        contents = self.dets[cnt]
        
        self.count += 1
        
        return contents
    
    def __len__(self):
        #test_print("len ==>",self.dets.__len__())
        return self.dets.__len__()
    
    def print(self, on=True):
        if on:
            print("==============================================")
            print(f'{self.vid}.{self.f_num}',"dets")
            for i, det in enumerate(self.dets):
                print(f'det {i} :', det)
            print("==============================================")
        
    def auto_set_dets(self):
        for i in range(randrange(1, 7)):
            det = {"frame": self.f_num, "x": randrange(10,1970), "y": randrange(10,1070), "w":randrange(18,31), "h":randrange(18,31), "cls":randrange(0,2), "conf":random()}
            self.dets.append(det)
    def set_det(self, idx=0, xywh=None, id=-1, cls=0, conf=0.0):
        #test_print(self.dets.__len__(), "det.len()")
        if self.dets.__len__() < idx:
            raise IndexError
        if xywh is not None:
            det = self.dets[idx]
            det["x"] = xywh[0]
            det["y"] = xywh[1]
            det["w"] = xywh[2]
            det["h"] = xywh[3]
            det["id"] = id
            det["cls"] = cls
            det["conf"] = conf

    def append_det(self, xywh, id=-1, cls=0, conf=0.0):
        det = dict()
        det["x"] = xywh[0]
        det["y"] = xywh[1]
        det["w"] = xywh[2]
        det["h"] = xywh[3]
        det["id"] = id
        det["cls"] = cls
        det["conf"] = conf
        self.dets.append(det)
            
    def imshow(self, name="no name", idx_names=["1","2","3"],cls_names=["E", "B"],line_thickness=2, hide_labels=True, hide_conf = True):
        im0= self.im0s
        annotator = Annotator(
            im0, line_width=line_thickness, example=str(cls_names))
        # Write results
        for i, det in enumerate(self.dets):
            c = int(det["cls"])  # integer class
            id = ""
            try:
               id = f"{idx_names[int(det['id'])]} "
            except KeyboardInterrupt:sys.exit()
            except:
                pass
            label = None if hide_labels else (f"{id}{cls_names[c]}" if hide_conf else f'{cls_names[c]} {det["conf"]:.2f}')
            xywh = [det["x"], det["y"], det["w"], det["h"]]
            xyxy = xywh2xyxy(xywh)
            annotator.box_label(xyxy, label, color=colors(c, True))
        im0 = annotator.result()
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 1280, 720)
        cv2.imshow(name, im0)
        # print(f"vid: {input.vid} / f_num: {input.f_num} / path: {input.path}")
        
    def is_detkey(self, key = "id") ->bool:
        try:
            for det in self.dets:
                a = det[key]
                test_print(f"is_detkey({key} : {a})")
        except KeyboardInterrupt:sys.exit()
        except:
            return False
        return True
    
    def len_unset_detkey(self, key = "id") -> int:
        try:
            return self.unset_det_num[key]
        except KeyboardInterrupt:sys.exit()
        except:
            cnt = 0
            for det in self.dets:
                try:
                    a = det[key]
                    test_print(f"is_detkey({key} : {a})")
                except:
                    cnt += 1
            self.unset_det_num[key] = cnt
            return self.unset_det_num[key]
    def len_detkey_match(self, key = "cls", value="1") -> int:
        cnt = 0
        if len(self.dets) <= 0:
            return 0
        
        for det in self.dets:
            if int(det[key]) == int(value):
                cnt += 1
        return cnt
    def update_id(self, key, value, xywh, conf, cls=1):
        for det in self.dets:
            if float(det["conf"]) == float(conf) and int(det["cls"]) == int(cls):
                if same_box([],[]):
                    det[key] = value
                        
def same_box(box1, box2, iou_th=0.9) -> bool:
    return True

def xywh2xyxy(x):
    y=list(x)
    y[0] = float(x[0]) - float(x[2]) / 2  # top left x
    y[1] = float(x[1]) - float(x[3]) / 2  # top left y
    y[2] = float(x[0]) + float(x[2]) / 2  # bottom right x
    y[3] = float(x[1]) + float(x[3]) / 2  # bottom right y
    return y

def xyxy2xywh(x):
    y=list(x)
    y[0] = (float(x[0]) + float(x[2])) / 2  # x center
    y[1] = (float(x[1]) + float(x[3])) / 2  # y center
    y[2] = float(x[2]) - float(x[0])  # width
    y[3] = float(x[3]) - float(x[1]) # height
    return y

def copy_piperesource(src:PipeResource)->PipeResource:
    dst = PipeResource()
    dst.vid = src.vid #video id / set undefind
    dst.f_num = src.f_num #video frame number / set undefind
    dst.dets = src.dets.copy()
    # get dateset  (path, im, im0s, vid_cap, s)
    dst.path = src.path        # video path
    dst.im = src.im.copy()      # image
    dst.im0s = src.im0s        # 
    dst.vid_cap = src.vid_cap  # video data
    dst.s =src.s              # string
    return dst


        
class IPipeHandler(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.det_idx2label = list()
    #change PipResource.dets using input data
    @abstractclassmethod
    def exe(self, input: PipeResource) -> PipeResource:#output
        pass

class IPipeObserver(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.b_pipe_name = ""
    #receive input from subject
    #update data OR execute handler.exe(input)
    @abstractclassmethod
    def push_src(self, input: PipeResource) -> None:
        pass
    
    def set_handler(self, handler:IPipeHandler) ->None:
        self.handler = handler
    
class IPipeSubject(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.next_pipe = None #plz Set IPipeObserver
        
        
    #output : register_type
    @abstractclassmethod
    def connect_pipe(self, pipe_observer: IPipeObserver) -> str:
        pass
    
    #output
    #   success fn -> true / fail fn -> false    
    @abstractclassmethod
    def unconnect_pipe(self, pipe_observer: IPipeObserver) -> bool:
        pass
    
    #work next_pipe.push_src(output)
    @abstractclassmethod
    def call_next_pipe(self, output : PipeResource, observer_idx=0) -> None:
        pass
    
    def unconnect_all(self):
        self.next_pipe = None
        
    @abstractclassmethod
    def get_regist_type(self, idx=0) -> str:
        pass
    
class IObserverPipe(IPipeHandler, IPipeObserver, IPipeSubject, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()
        self.set_handler(self)

    #output : register_type
    @abstractclassmethod
    def connect_pipe(self, pipe_observer: IPipeObserver) -> str:
        pass
    
    #output
    #   success fn -> true / fail fn -> false    
    @abstractclassmethod
    def unconnect_pipe(self, pipe_observer: IPipeObserver) -> bool:
        pass
    
    #work next_pipe.push_src(output)
    @abstractclassmethod
    def call_next_pipe(self, output : PipeResource, observer_idx=0) -> None:
        pass
    
    @abstractclassmethod
    def push_src(self, input: PipeResource) -> None:
        pass
    
    @abstractclassmethod
    def exe(self, input: PipeResource) -> PipeResource:#output
        pass
    @abstractclassmethod
    def get_regist_type(self, idx=0) -> str:
        pass
    
class IOne2OnePipe(IObserverPipe, metaclass=ABCMeta):
    def __init__(self) -> None :
        super().__init__()
        test_print(self.__str__(),"pipe_cls test - one2one init")
    
    @abstractclassmethod
    def push_src(self, input: PipeResource) -> None :
        pass
        
    @abstractclassmethod
    def exe(self, input: PipeResource) -> PipeResource :#output
        pass
    @abstractclassmethod
    def get_regist_type(self, idx=0) -> str:
        pass
      
    def connect_pipe(self, pipe_observer: IPipeObserver) -> str:
        if isinstance(pipe_observer, IPipeObserver):
            test_print(self.__str__() + f" connect_pipe({pipe_observer.__str__()})")
            self.next_pipe = pipe_observer                      #Set next pipe
            pipe_observer.b_pipe_name = self.get_regist_type()
            return self.get_regist_type()                       #rerutn type
        return "Not_IPipeObserver"
    
    def unconnect_pipe(self, pipe_observer: IPipeObserver) -> bool:
        if isinstance(pipe_observer, IPipeObserver):
            if pipe_observer == self.next_pipe:
                test_print(self.__str__() + " unconnect_pipe")
                self.next_pipe = None                           #delete pipeline
                return True
        return False
    
    def call_next_pipe(self, output : PipeResource, observer_idx=0) -> None:
        try:
            if isinstance(output, PipeResource):                                #check input valid
                if isinstance(self.next_pipe, IPipeObserver):                   #check next pipe valid
                    test_print(self.__str__() + f" call_next_pipe({self.next_pipe.__str__()})")
                    self.next_pipe.push_src(output)
        except KeyboardInterrupt:sys.exit()
        except NotEnoughDetectError as ex: raise NotEnoughDetectError(str(ex))
        except Exception as ex:
            print(self.__str__(), " One2OnePipe call_next_pipe : ", ex.__str__())
    
    
class One2OnePipe(IOne2OnePipe, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()
            
    def push_src(self, input: PipeResource) -> None:
        if input is not None:
            if isinstance(input, PipeResource):                                #check input valid
                # test_print(self.__str__() + " puch_src")
                t1 = time.time()
                output = self.handler.exe(input)
                t2 = time.time()
                test_print(f"[{self.get_regist_type()} exe {t2-t1}s]",end=" ")
                if output is not None:                                          #no action condition
                    self.call_next_pipe(output)
        
    @abstractclassmethod
    def exe(self, input: PipeResource) -> PipeResource:#output
        pass
    @abstractclassmethod
    def get_regist_type(self, idx=0) -> str:
        pass

class One2ManyPipe(IObserverPipe, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()
        test_print("pipe_cls test - one2many init")
        self.next_pipe = list()
        self.idx2label = list()
        
        
    def connect_pipe(self, pipe_observer: IPipeObserver) -> str:
        if isinstance(pipe_observer, IPipeObserver):
            test_print(self.__str__() + f" connect_pipe({pipe_observer.__str__()})")
            self.next_pipe.append(pipe_observer)                            #Set next pipe
            pipe_observer.b_pipe_name = self.get_regist_type(idx=self.next_pipe.__len__()-1)
            return self.get_regist_type(idx=self.next_pipe.__len__()-1)       #rerutn type
        return "Not_IPipeObserver"
    
    def unconnect_pipe(self, pipe_observer: IPipeObserver) -> bool:
        if isinstance(pipe_observer, IPipeObserver):
            target_observer_idx = -1
            #find observer
            try:
                self.next_pipe.remove(pipe_observer)                        #delete pipeline
                return True
            except KeyboardInterrupt:sys.exit()
            except:
                pass
        return False
    
    def call_next_pipe(self, output : PipeResource, observer_idx=0) -> None:
        if isinstance(output, PipeResource):                                #check input valid
            try:
                if isinstance(self.next_pipe[observer_idx], IPipeObserver):     #check next pipe valid
                    # test_print(self.__str__() + " call_next_pipe")
                    self.next_pipe[observer_idx].push_src(output)
            except KeyboardInterrupt:sys.exit()
            except NotEnoughDetectError as ex: raise NotEnoughDetectError(str(ex))
            except IndexError as ex:
                pass
                #print(self.__str__(), f" call_next_pipe({observer_idx}/{self.next_pipe.__len__()}) : ", ex.__str__())
            except Exception as ex:
                print(self.__str__(), " One2ManyPipe call_next_pipe : ", ex.__str__())
           
    @abstractclassmethod
    def push_src(self, input: PipeResource) -> None:
        pass
        
    @abstractclassmethod
    def exe(self, input: PipeResource) -> PipeResource:#output
        pass
    @abstractclassmethod
    def get_regist_type(self, idx=0) -> str:
        pass
    
class RepeatPipe(One2ManyPipe, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()
        
    def push_src(self, input: PipeResource) -> None:
        if isinstance(input, PipeResource):                                #check input valid
            # test_print(self.__str__() + " puch_src")
            output = self.handler.exe(input)
            for i in range(self.next_pipe.__len__()):
                self.call_next_pipe(output=output, observer_idx=i)
        
    @abstractclassmethod
    def exe(self, input: PipeResource) -> PipeResource:#output
        pass
    @abstractclassmethod
    def get_regist_type(self, idx=0) -> str:
        pass

class SplitPipe(One2ManyPipe, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()
    
    def push_src(self, input: PipeResource) -> None:
        if isinstance(input, PipeResource):                                #check input valid
            # test_print(self.__str__() + " puch_src")
            output = self.handler.exe(input)
            for i, out in enumerate(output.dets):
                self.call_next_pipe(output=out, observer_idx=i)
        
    #plz Return splited output
    @abstractclassmethod
    def exe(self, input: PipeResource) -> PipeResource:
        pass
    @abstractclassmethod
    def get_regist_type(self, idx=0) -> str:
        pass
    def create_test_resource(f_num, im=None, im0s=None) -> PipeResource:    
            src = PipeResource(vid=1, f_num=f_num, im=im, im0s=im0s, s="test data")
            src.auto_set_dets()
            return src        

class StartNotDetectCutterPipe(One2OnePipe):
    def __init__(self) -> None:
        super().__init__()
        self.frame_num = 0
    def exe(self, input: PipeResource) -> PipeResource:
        if self.frame_num ==0:
            if input.dets.__len__() < 3:
                input = None
            elif input.dets.__len__() >=4 :
                class OverStartDetectError(Exception):
                    def __init__(self, msg = "Over Detect in start frame"):
                        super().__init__(msg)
                raise OverStartDetectError()
            else :
                self.frame_num += 1
        return input
    
    def get_regist_type(self, idx=0) -> str:
        return "start cutter"

class FirstCopyPipe(IOne2OnePipe):
    def __init__(self, N_INIT=30, display=True) -> None:
        super().__init__()
        self.display = display
        self.n_init = N_INIT
        self.frame_num = 0
            
    def push_src(self, input: PipeResource) -> None:
        
        if input is not None and len(input.dets) > 0:
            if isinstance(input, PipeResource):                                #check input valid
                # test_print(self.__str__() + " puch_src")
                output = self.handler.exe(input)
                
                t1 = time.time()
                if self.frame_num == 0:
                    if output is not None:                                          #no action condition
                        k=self.n_init
                        for i in range(k-1):
                            self.call_next_pipe(output)
                        self.frame_num += 1
                    t2 = time.time()
                    if self.display and k != 1:
                        print(f"[{self.get_regist_type()} {t2-t1:.3f}s]",end=" ")
                self.call_next_pipe(output)
                
                    
    
    def exe(self, input: PipeResource) -> PipeResource:
        return input
    
    def get_regist_type(self, idx=0) -> str:
        return "FirstCopy"

class ConvertToxywhPipe(One2OnePipe):
    def __init__(self) -> None:
        super().__init__()
        
    def exe(self, input: PipeResource) -> PipeResource:
        dets = input.dets
        for det in dets:
            xyxy = [det["x"],det["y"],det["w"],det["h"]]
            xywh = xyxy2xywh(xyxy)
            det["x"] = xywh[0]
            det["y"] = xywh[1]
            det["w"] = xywh[2]
            det["h"] = xywh[3]
        # test_print("++++++++++++++convert xywh+++++++++++++++++")
        # input.print(on=is_test())
        return input
    
    def get_regist_type(self, idx=0) -> str:
        return "xyxy2xywh"


class ResourceOne(IPipeObserver):
    def __init__(self, cls_type="") -> None:
        super().__init__()
        self.src=None
        self.b_pipe_name = cls_type
        
    def push_src(self, input: PipeResource) -> None:
        self.src = input
        
    def print(self):
        self.src.print()

class ResourceBag(IPipeObserver):
    def __init__(self, cls_type="") -> None:
        super().__init__()
        self.src_list = list()
        self.b_pipe_name = cls_type
    
    def __len__(self):
        cnt = 0
        for ball in self.src_list:
            if ball.__len__() > 0 :
                cnt += 1
        return cnt
    
    def push_src(self, input: PipeResource) -> None:
        self.src_list.append(input)
    
    def print(self, on=True):
        if on:
            print(f"=======================[type({self.b_pipe_name})]======================")
            for i, src in enumerate(self.src_list):
                print("idx:",i, end=" ")
                src.print()
            
    def print_last(self):
        test_print(f"len : {self.src_list.__len__()}")
        print(f"[type({self.b_pipe_name})/idx({self.src_list.__len__()})]")
        self.src_list[-1].print()
        
    #return [dict, dict, ... ,dict]
    def get_list(self)->list:
        contents = []
        
        for pipe_src in self.src_list:
            for src in pipe_src:
                contents.append(src)
        
        return contents


class CopyPipe(RepeatPipe):
    def __init__(self) -> None:
        super().__init__()
    
    def exe(self, input: PipeResource) -> PipeResource:
        return input
    
    def get_regist_type(self, idx=0) -> str:
        return "copy_pipe"+idx.__str__()



class SplitKey(SplitPipe):
    def __init__(self, names=[], key="cls") -> None:
        super().__init__()
        for name in names:
            self.idx2label.append(name)
        self.target_detkey = key
        
    def exe(self, input: PipeResource) -> PipeResource:
        output = PipeResource()
        try:
            #setting output 
            for i in range(self.idx2label.__len__()):
                o = copy_piperesource(input)
                o.dets.clear()
                output.dets.append(o)
        except KeyboardInterrupt:sys.exit()
        except Exception as ex:
            print(self.__str__(), " exe(setting) : ", ex.__str__()) 
            return input
        #split det
        for det in input.dets:
            try:
                value = int(det[self.target_detkey])
                # test_print("exe", value)
                output.dets[value].dets.append(det)
            except KeyboardInterrupt:sys.exit()
            except Exception as ex:
                test_print(self.__str__(), " exe : ", ex.__str__()) 
        # test_print("=============split key=============")
        # for o in output.dets:
        #     o.print(on=is_test())
        return output
    
    def get_regist_type(self, idx=0) -> str:
        return self.idx2label[idx]
class SplitIdx(SplitKey):
    def __init__(self) -> None:
        super().__init__(["1", "2", "3", "4", "5", "6"], "id")
class SplitCls(SplitKey):
    def __init__(self) -> None:
        super().__init__(["EDGE", "BALL"], "cls")

#########################==================test case=============================

class PassPipe(One2OnePipe):
    def __init__(self) -> None:
        super().__init__()
        self.istest = is_test()
    
    def exe(self, input: PipeResource) -> PipeResource:
        if self.istest:
            try:
                for i, det in enumerate(input.dets):
                    det["test"] = True
            except KeyboardInterrupt:sys.exit()
            except:
                test_print("error : PassPipe")
        return input
    
    def get_regist_type(self, idx=0) -> str:
        return "pass_pipe"

  
def test_split_pipe():
    print("========Run exe============")
    src = create_test_resource(f_num=0)
    print("input")
    src.print()
    
    p_pipe = SplitCls()
    output = p_pipe.exe(src)
    print("exe output")
    output.print()
    output.dets[0].print()
    output.dets[1].print()
    
    print("========connect bag1 and push_src============")
    src = create_test_resource(f_num=0)
    print("input")
    src.print()
    
    bag1 = ResourceBag()
    p_pipe.connect_pipe(bag1)
    
    p_pipe.push_src(src)
    print("push_src output")
    bag1.print_last()
    
    print("========connect bag2 and push_src============")
    src = create_test_resource(f_num=1)
    print("input")
    src.print()
    
    bag2 = ResourceBag()
    p_pipe.connect_pipe(bag2)
    
    p_pipe.push_src(src)
    print("push_src output")
    bag1.print_last()
    bag2.print_last()
    
    print("========Repeat push_src============")
    for i in range(2, randrange(3,10)):
        src = create_test_resource(f_num=i)
        print(f"frame{i} : input")
        src.print()
        
        p_pipe.push_src(src)
        print(f"frame{i} : push_src output")
        #bag.print_all()
        bag1.print_last()
        bag2.print_last()
    bag1.print_all()
    bag2.print_all()

def test_repeat_pipe():
    print("========Run exe============")
    src = create_test_resource(f_num=0)
    print("input")
    src.print()
    
    p_pipe = CopyPipe()
    output = p_pipe.exe(src)
    print("exe output")
    output.print()

    print("========connect bag1 and push_src============")
    src = create_test_resource(f_num=1)
    print("input")
    src.print()
    
    bag1 = ResourceBag()
    p_pipe.connect_pipe(bag1)
    
    p_pipe.push_src(src)
    print("push_src output")
    bag1.print_last()
    
    print("========connect bag2 and push_src============")
    src = create_test_resource(f_num=1)
    print("input")
    src.print()
    
    bag2 = ResourceBag()
    p_pipe.connect_pipe(bag2)
    
    p_pipe.push_src(src)
    print("push_src output")
    bag1.print_last()
    bag2.print_last()
    
    print("========Repeat push_src============")
    for i in range(2, randrange(3,10)):
        src = create_test_resource(f_num=i)
        print(f"frame{i} : input")
        src.print()
        
        p_pipe.push_src(src)
        print(f"frame{i} : push_src output")
        #bag.print_all()
        bag1.print_last()
        bag2.print_last()
    bag1.print_all()
    bag2.print_all()


def test_o2opipe():
    
    print("========Run exe============")
    src = create_test_resource(f_num=0)
    print("input")
    src.print()
    
    p_pipe = PassPipe()
    output = p_pipe.exe(src)
    print("exe output")
    output.print()

    print("========Run push_src============")
    src = create_test_resource(f_num=1)
    print("input")
    src.print()
    
    bag = ResourceBag()
    p_pipe.connect_pipe(bag)
    
    p_pipe.push_src(src)
    print("push_src output")
    bag.print_last()
    
    print("========Repeat push_src============")
    for i in range(2, randrange(3,10)):
        src = create_test_resource(f_num=i)
        print(f"frame{i} : input")
        src.print()
        
        p_pipe.push_src(src)
        print(f"frame{i} : push_src output")
        #bag.print_all()
        bag.print_last()


if __name__ == '__main__':
    #test_o2opipe()
    #test_repeat_pipe()
    test_split_pipe()