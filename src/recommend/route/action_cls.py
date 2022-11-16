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
    def get_distance_from_point(x:float, y:float)-> float:
        pass
    
    # 점에 대한 법선 벡트를 반환한다.
    @abstractclassmethod
    def get_normal_vector(self, x:float, y:float)-> np.array:
        pass
    
    @abstractclassmethod
    def crash(self, normal_vec:np.array):
        pass

class CaromBall(IMoveable, ICrashable, IObserver, ISubject):
    def __init__(self) -> None:
        super().__init__()
        
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
    
    def update(self, event:dict=None) -> None:
        pass

    def notify_observers(self):
        pass

    def get_distance_from_point(x:float, y:float)-> float:
        pass

    def get_normal_vector(self, x:float, y:float)-> np.array:
        pass
    
    def crash(self, normal_vec:np.array):
        pass