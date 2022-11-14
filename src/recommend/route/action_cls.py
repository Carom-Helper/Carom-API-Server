from abc import *
import numpy as np

from route_utills import is_test

def is_test_action_cls()->bool:
    return False and is_test()

def test_print(s, s1="", s2="", s3="", s4="", s5="", end="\n"):
    if is_test_action_cls():
        print("action cls exe : ", s, s1, s2, s3, s4, s5, end=end)

class IObserver(metaclass=ABCMeta):
    @abstractclassmethod
    def update(self, time=None, reflect=None) -> None:
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
    def __init__(self) -> None:
        self.init()
    
    def init(self):
        self.vec = np.array([0,0]) # 방향벡터
        self.pos = np.array([0,0]) # 현재 위치
        self.speed = np.array([0]) # 
        
    # 해당 시간이 지날 때 거리를 반환다.
    @abstractclassmethod
    def move(self, t:float)->float:
        pass
    
class ICrashable(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass
    
    # x,y 점과 자신과의 거리를 반환한다.
    @abstractclassmethod
    def get_distance_from_point(x:float, y:float)-> float:
        pass
    
    # 점에 대한 법선 벡트를 반환한다.
    @abstractclassmethod
    def get_reflect_vector(self, x:float, y:float)-> np.array:
        pass
    
    @abstractclassmethod
    def crash(self, reflect_vec:np.array):
        pass

class CaromBall(IMoveable, ICrashable, IObserver, ISubject):
    def __init__(self) -> None:
        super().__init__()
        
    def set_vec(self, vec:np.array):
        self.vec = vec
    
    