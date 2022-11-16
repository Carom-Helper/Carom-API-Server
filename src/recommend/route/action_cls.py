from abc import *
import numpy as np
import math

from route_utills import is_test

def is_test_action_cls()->bool:
    return False and is_test()

def test_print(s, s1="", s2="", s3="", s4="", s5="", end="\n"):
    if is_test_action_cls():
        print("action cls exe : ", s, s1, s2, s3, s4, s5, end=end)

class IObserver(metaclass=ABCMeta):
    @abstractclassmethod
    def update(self, event:dict=None) -> None:
        pass

class ISubject(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.observer_list=list()    
        
    def register_observer(self, observer:IObserver) -> int:
        if isinstance(observer, IObserver):
            self.observer_list.append(observer)
            return len(self.observer_list)
        return None
    
    def remove_observer(self, observer:IObserver) -> bool:
        if isinstance(observer, IObserver):
            try:
                self.observer_list.remove(observer)
                return True
            except ValueError as ex:
                test_print("remove observer", str(ex))
        return False
    
    @abstractclassmethod
    def notify_observers(self):
        pass

class IMoveable(metaclass=ABCMeta):
    # mover is closure
    def set_mover(self, mover) ->None:
        self.mover = mover
    # 해당 시간이 지날 때 거리를 반환다.
    def move(self, t:float)->float:
        return self.mover(t)
    @abstractclassmethod
    def get_xy(self)->list:
        pass
    
# class SaveDistanceMoveAble(IMoveable):
#     def __init__(self) -> None:
#         super().__init__()
        
#     def move(self, t: float) -> float:
#         distance = super().move(t)
#         self.distance =+ distance
#         return distance
    
class ICrashable(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass
    
    # x,y 점과 자신과의 거리를 반환한다.
    @abstractclassmethod
    def get_distance_from_point(self, x:float, y:float)-> float:
        pass
    
    # 점에 대한 법선 벡트를 반환한다.
    @abstractclassmethod
    def get_normal_vector(self, x:float, y:float)-> np.array:
        pass
    
    # 점에 대한 법선 벡트를 반환한다.
    @abstractclassmethod
    def get_reflect_closure(self):
        pass
    
class ICrashAction(metaclass=ABCMeta):
    # 충돌 했을 때 발생하는 이벤트를 받는다. 
    # crasher is closure
    @abstractclassmethod
    def crash(self, crashable:ICrashable):
        pass

class ICrash(ICrashable, ICrashAction, metaclass=ABCMeta):
    # x,y 점과 자신과의 거리를 반환한다.
    @abstractclassmethod
    def get_distance_from_point(self, x:float, y:float)-> float:
        pass
    # 점에 대한 법선 벡트를 반환한다.
    @abstractclassmethod
    def get_normal_vector(self, x:float, y:float)-> np.array:
        pass
    # 점에 대한 법선 벡트를 반환한다.
    @abstractclassmethod
    def get_reflect_closure(self):
        pass
    # 충돌 했을 때 발생하는 이벤트를 받는다. 
    # crasher is closure
    @abstractclassmethod
    def crash(self, crashable:ICrashable):
        pass

class CaromBall(IMoveable, ICrashable, IObserver, ISubject):
    def __init__(self) -> None:
        super().__init__()
        self.xy = []
        self.vector = {"x": 0, "y": 0}
        
    def start_param(self, power = 50, clock = 12, tip = 0):
        radius = 8.6
        self.power = power
        self.theta = clock % 12 * (-30) + 90
        self.tip = tip
        upspinmax = 3  * math.sin(math.pi * (90 / 180)) * 50 * radius
        upspinmin = 3  * math.sin(math.pi * (-60 / 180)) * 50 * radius
        self.upspin = math.sin(math.pi * (self.theta/180)) * tip * self.power * radius
        upspinrate = int((self.upspin - upspinmin) / (upspinmax-upspinmin) * 10)

        self.upspinrate = upspinrate

        sidespinmax = 3 * math.cos(math.pi * (0 / 180)) * 50 * radius
        sidespinmin = 3 * math.cos(math.pi * (-180 / 180)) * 50 * radius
        self.sidespin = math.cos(math.pi * (self.theta/180)) * tip * self.power * radius
        sidespinrate = int((self.sidespin - sidespinmin) / (sidespinmax-sidespinmin) * 10)

        self.sidespinrate = sidespinrate-5
    
    def print_param(self):
        print(f'theta: {self.theta}, tip: {self.tip}/3')
        print(f'upspin: {self.upspin:0.2f}, sidespin: {self.sidespin:0.2f}')
        print(f'upspinrate: {self.upspinrate}, sidespinrate: {self.sidespinrate}\n')
    
    @abstractclassmethod
    def get_range(self, data=dict())->float:
        pass
    @abstractclassmethod
    def get_xy(self)->list:
        pass
    
    
class CrashableSubject(ICrashable, ISubject, meta=ABCMeta):
    def __init__(self) -> None:
        super().__init__()
        self.crash_event = None
    
    def set_crash_event(self, crash_action:ICrashAction):
        self.crash_event = crash_action
    
    @abstractclassmethod
    def notify_observers(self):
        pass
        # # observer들을 하나씩 방문하면서
        # for idx, observer in enumerate(self.observer_list):
        #     # 충돌 인지를 판정한다.
        #     if self.check_crash(idx):
        #         # 충돌 했다면, 충돌을 전파한다.
        #         observer.crash(self)
        #         # 그리고 자기도 충돌 행동을 한다. 하지만 안한다.
    # x,y 점과 자신과의 거리를 반환한다.
    @abstractclassmethod
    def get_distance_from_point(self, x:float, y:float)-> float:
        pass
    # 점에 대한 법선 벡트를 반환한다.
    @abstractclassmethod
    def get_normal_vector(self, x:float, y:float)-> np.array:
        pass
    # 점에 대한 법선 벡트를 반환한다.
    @abstractclassmethod
    def get_reflect_closure(self):
        pass

    def get_xy(self)->list:
        return self.xy

    def set_xy(self, x:float, y:float):
        temp = {"x": x, "y": y, "elapsed": 0}
        self.xy.append(temp)

    def add_xy(self, xy:dict):
        self.xy.append(xy)

    def move(self, t:float)->float:
        xy = self.mover(t)
        self.add_xy(xy)
        self.notify_observers()
        return 

    def move_by_time(self, elapsed:float)->float:
        x, y = self.xy[-1]["x"], self.xy[-1]["y"]
        new_x, new_y = x + self.vector["x"] * elapsed, y + self.vector["y"] * elapsed

        xy = {"x": new_x, "y": new_y, "t": self.xy[-1]["elapsed"] + elapsed}

        return xy

class IMovableObserver(IMoveable, IObserver):
    def __init__(self) -> None:
        super().__init__()
    @abstractclassmethod
    def get_xy(self)->list:
        pass
    @abstractclassmethod
    def update(self, event:dict=None) -> None:
        pass


def set_vec(cue:CaromBall, tar:CaromBall, thickness:float)->dict:
    radius = 8.6

    cue_pos = cue.get_xy()[-1]
    tar_pos = tar.get_xy()[-1]

    cue_tar = {'x':(cue_pos['x'] - tar_pos['x']), 'y':(cue_pos['y'] - tar_pos['y'])}
    new_x = thickness/8 * radius
    new_y = (radius**2 - new_x**2)**0.5

    new_x *= 1.5
    new_y *= 1.5

    new_t = {'x': new_x, 'y': new_y}

    cue_tar_l = (cue_tar['x']**2+cue_tar['y']**2)**0.5
    cos = cue_tar['y'] / cue_tar_l
    sin = -cue_tar['x'] / cue_tar_l

    new_t['x'] = new_x * cos - new_y * sin + tar_pos['x']
    new_t['y'] = new_x * sin + new_y * cos + tar_pos['y']

    vector = {"x": new_t["x"] - cue_pos["x"], "y": new_t["y"] - cue_pos["y"]}

    length = (vector["x"]**2 + vector["y"]**2)**0.5
    vector["x"] *= 3/5 / length * cue.power / 50
    vector["y"] *= 3/5 / length * cue.power / 50
    
    cue.vector["x"] = vector["x"]
    cue.vector["y"] = vector["y"]

class ICrashObserver(ICrash, IObserver, meta=ABCMeta):
    def __init__(self) -> None:
        super().__init__()
    # x,y 점과 자신과의 거리를 반환한다.
    @abstractclassmethod
    def get_distance_from_point(self, x:float, y:float)-> float:
        pass
    # 점에 대한 법선 벡트를 반환한다.
    @abstractclassmethod
    def get_normal_vector(self, x:float, y:float)-> np.array:
        pass
    # 점에 대한 법선 벡트를 반환한다.
    @abstractclassmethod
    def get_reflect_closure(self):
        pass
    # 충돌 했을 때 발생하는 이벤트를 받는다. 
    # crasher is closure
    @abstractclassmethod
    def crash(self, crashable:ICrashable):
        pass
    @abstractclassmethod
    def update(self, event:dict=None) -> None:
        pass


class MoveableSubject(IMoveable, ISubject, meta=ABCMeta):
    elapse = 0.001
    def __init__(self) -> None:
        super().__init__()
        self.crashable = None
    
    def set_crashable(self, crashable:ICrashable):
        if crashable is not ICrashable:raise TypeError
        self.crashable = crashable
    
    def register_observer(self, observer: ICrashObserver) -> int:
        if observer is not ICrashObserver: raise TypeError
        return super().register_observer(observer)
    
    def check_crash(self, observer:ICrash):
        x,y = self.get_xy()
        distance = observer.get_distance_from_point(x,y)
        return (distance - self.get_range()) < self.elapse
    
    # 자신과 충돌한 것이 있는지 없는지 확인한다. 
    # 그리고 충돌한 것이 있다면, 발견한 즉시 검색을 멈추고 충돌을 전파한다.
    # 전파 후에는 충돌한 객체 정보를 반환한다. 충돌한 것이 없으면 None 반환
    def notify_observers(self)-> ICrashObserver:
        # observer들을 하나씩 방문하면서
        target_observer = None
        for idx, observer in enumerate(self.observer_list):
            # 충돌 인지를 판정한다.
            if self.check_crash(idx):
                # 충돌 했다면, 타겟 지정
                target_observer = observer
                break
        # 아무것도 없으면, 종류
        if target_observer is None: return None
        
        # 충돌 확인
        if self.crashable is None: raise TypeError(f"{str(self)}.+ plz set_crashable(self, crashable:ICrashable)")
        
        # 충돌 했다면, 충돌을 전파한다.
        observer.crash(self.crashable)
        # 그리고 자기도 충돌 행동을 한다. 하지만 안한다.
        # 나중에 super를 사용한 뒤에 알아서 구현해라
        return observer
    
    @abstractclassmethod
    def get_range(self, data=dict())->float:
        pass
    @abstractclassmethod
    def get_xy(self)->list:
        pass
    
    
class CrashableSubject(ICrashable, ISubject, meta=ABCMeta):
    def __init__(self) -> None:
        super().__init__()
        self.crash_event = None
    
    def set_crash_event(self, crash_action:ICrashAction):
        self.crash_event = crash_action
    
    @abstractclassmethod
    def notify_observers(self):
        pass
        # # observer들을 하나씩 방문하면서
        # for idx, observer in enumerate(self.observer_list):
        #     # 충돌 인지를 판정한다.
        #     if self.check_crash(idx):
        #         # 충돌 했다면, 충돌을 전파한다.
        #         observer.crash(self)
        #         # 그리고 자기도 충돌 행동을 한다. 하지만 안한다.
    # x,y 점과 자신과의 거리를 반환한다.
    @abstractclassmethod
    def get_distance_from_point(self, x:float, y:float)-> float:
        pass
    # 점에 대한 법선 벡트를 반환한다.
    @abstractclassmethod
    def get_normal_vector(self, x:float, y:float)-> np.array:
        pass
    # 점에 대한 법선 벡트를 반환한다.
    @abstractclassmethod
    def get_reflect_closure(self):
        pass


class IMovableObserver(IMoveable, IObserver):
    def __init__(self) -> None:
        super().__init__()
    @abstractclassmethod
    def get_xy(self)->list:
        pass
    @abstractclassmethod
    def update(self, event:dict=None) -> None:
        pass