from abc import *

from random import randrange
import time
import sys

from detect_utills import PipeResource, xyxy2xywh, copy_piperesource, is_test
def is_test_pipe_cls()->bool:
    return False and is_test()

def test_print(s, s1="", s2="", s3="", s4="", s5="", end="\n"):
    if is_test_pipe_cls():
        print("pipe cls exe : ", s, s1, s2, s3, s4, s5, end=end)



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
        except Exception as ex:
            print(f'{str(self)} One2OnePipe call_next_pipe : {str(ex)}')
    
    
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
    
def create_test_resource(im=None, images=dict(), metadata=dict()) -> PipeResource:
    src = PipeResource(metadata=metadata, images=images, im=im)
    src.auto_set_dets()
    return src     

class ConvertToxywhPipe(One2OnePipe):
    def __init__(self) -> None:
        super().__init__()
        
    def exe(self, input: PipeResource) -> PipeResource:
        dets = input.dets
        for det in dets:
            xyxy = [det["xmin"],det["ymin"],det["xmax"],det["ymax"]]
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
    def __init__(self, display=True) -> None:
        super().__init__()
        self.display = display
    
    def exe(self, input: PipeResource) -> PipeResource:
        if self.display:
            print(f'{input.s}')
        return input
    
    def get_regist_type(self, idx=0) -> str:
        return "pass_pipe"


  
def test_split_pipe():
    print("========Run exe============")
    src = create_test_resource({"f_num":0})
    print("input")
    src.print()
    
    p_pipe = SplitCls()
    output = p_pipe.exe(src)
    print("exe output")
    try:
        output.print()
    except:
        output.dets[0].print()
        output.dets[1].print()
    
    print("========connect bag1 and push_src============")
    src = create_test_resource({"f_num":0})
    print("input")
    src.print()
    
    bag1 = ResourceBag()
    p_pipe.connect_pipe(bag1)
    
    p_pipe.push_src(src)
    print("push_src output")
    bag1.print_last()
    
    print("========connect bag2 and push_src============")
    src = create_test_resource({"f_num":1})
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
        src = create_test_resource({"f_num":i})
        print(f"frame{i} : input")
        src.print()
        
        p_pipe.push_src(src)
        print(f"frame{i} : push_src output")
        #bag.print_all()
        bag1.print_last()
        bag2.print_last()
    bag1.print()
    bag2.print()

def test_repeat_pipe():
    print("========Run exe============")
    src = create_test_resource({"f_num":0})
    print("input")
    src.print()
    
    p_pipe = CopyPipe()
    output = p_pipe.exe(src)
    print("exe output")
    output.print()

    print("========connect bag1 and push_src============")
    src = create_test_resource({"f_num":1})
    print("input")
    src.print()
    
    bag1 = ResourceBag()
    p_pipe.connect_pipe(bag1)
    
    p_pipe.push_src(src)
    print("push_src output")
    bag1.print_last()
    
    print("========connect bag2 and push_src============")
    src = create_test_resource({"f_num":1})
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
        src = create_test_resource({"f_num":i})
        print(f"frame{i} : input")
        src.print()
        
        p_pipe.push_src(src)
        print(f"frame{i} : push_src output")
        #bag.print_all()
        bag1.print_last()
        bag2.print_last()
    bag1.print()
    bag2.print()


def test_o2opipe():
    
    print("========Run exe============")
    src = create_test_resource({"f_num":0})
    print("input")
    src.print()
    
    p_pipe = PassPipe()
    output = p_pipe.exe(src)
    print("exe output")
    output.print()

    print("========Run push_src============")
    src = create_test_resource({"f_num":1})
    print("input")
    src.print()
    
    bag = ResourceBag()
    p_pipe.connect_pipe(bag)
    
    p_pipe.push_src(src)
    print("push_src output")
    bag.print_last()
    
    print("========Repeat push_src============")
    for i in range(2, randrange(3,10)):
        src = create_test_resource({"f_num":i})
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