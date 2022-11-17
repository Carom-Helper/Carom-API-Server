from abc import *
import numpy as np

from route_utills import is_test

def is_test_action_cls()->bool:
    return True and is_test()

def test_print(s, s1="", s2="", s3="", s4="", s5="", end="\n"):
    if is_test_action_cls():
        print("action cls exe : ", s, s1, s2, s3, s4, s5, end=end)

class IObserver(metaclass=ABCMeta):
    @abstractclassmethod
    def update(self, event:dict=None) -> None:
        pass

class IFitteringNotifier(metaclass=ABCMeta):
    @abstractclassmethod
    def notify_filltered_observer(self, observer:IObserver)->None:
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
    def notify_observers(self)->None:
        pass

class IFitteringSubject(IFitteringNotifier, ISubject, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()
    @abstractclassmethod
    def notify_filltered_observer(self, observer:IObserver)->None:
        pass
    
    def notify_observers(self)->None:
        for observer in self.observer_list:
            self.notify_filltered_observer(observer)
        return observer
    

class IMoveable(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.mover = None
    # mover is closure
    def set_mover(self, mover) ->None:
        self.mover = mover
    # 해당 시간이 지날 때 거리를 반환다.
    def move(self, t:float)->float:
        if self.mover is None: raise TypeError("Set Mover")
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
    
    # 점에 대한 법선 벡트를 반환하는 클로저를 반환한다.
    @abstractclassmethod
    def get_reflect_closure(self, direct_vec, normal_vec, power):
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
    def get_reflect_closure(self, direct_vec, normal_vec, power):
        pass
    # 충돌 했을 때 발생하는 이벤트를 받는다. 
    # crasher is closure
    @abstractclassmethod
    def crash(self, crashable:ICrashable):
        pass

class ICrashObserver(ICrashAction, ICrashable, IObserver, metaclass=ABCMeta):
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
    def get_reflect_closure(self, direct_vec, normal_vec, power):
        pass
    # 충돌 했을 때 발생하는 이벤트를 받는다. 
    # crasher is closure
    @abstractclassmethod
    def crash(self, crashable:ICrashable):
        pass
    @abstractclassmethod
    def update(self, event:dict=None) -> None:
        pass

class ICrashChecker(ICrashAction, metaclass=ABCMeta):
    elapse = 0.0001
    def __init__(self) -> None:
        super().__init__()
        
    # 자신과 충돌한 것이 있는지 없는지 확인한다. 
    def check_crash(self, observer:ICrash, x, y)-> bool:
        if observer is not ICrash: raise TypeError()
        distance = observer.get_distance_from_point(x,y)
        return ((distance - self.get_range()) < self.elapse)
    
    @abstractclassmethod
    def get_range(self, data=dict())->float:
        pass
    @abstractclassmethod
    def crash(self, crashable:ICrashable):
        pass
        # # observer들을 하나씩 방문하면서
        # target_observer = None
        # for idx, observer in enumerate(self.observer_list):
        #     # 충돌 인지를 판정한다.
        #     if self.check_crash(idx):
        #         # 충돌 했다면, 타겟 지정
        #         target_observer = observer
        #         break
        # # 아무것도 없으면, 종류
        # if target_observer is None: return None
        
        # # 충돌 확인
        # if self.crashable is None: raise TypeError(f"{str(self)}.+ plz set_crashable(self, crashable:ICrashable)")
        
        # # 충돌 했다면, 충돌을 전파한다.
        # observer.crash(self.crashable)
        # # 그리고 자기도 충돌 행동을 한다. 하지만 안한다.
        # # 나중에 super를 사용한 뒤에 알아서 구현해라



class IMoveableSubject(IMoveable, IFitteringSubject, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractclassmethod
    def get_xy(self)->list:
        pass
    
    @abstractclassmethod
    def notify_filltered_observer(self, observer:IObserver)->None:
        pass
    
class ICrashableSubject(ICrashable, IFitteringSubject, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()

    @abstractclassmethod
    def notify_filltered_observer(self, observer:IObserver)->None:
        pass
    
    # 접선의 방정식을 구하고
    # 수선의 발에서 그은 x,y 값과의 거리를 반환한다.
    @abstractclassmethod
    def get_distance_from_point(self, x:float, y:float)-> float:
        pass
    # 접선의 방정식을 구하고
    # 접선의 방정식의 수직하는 방정식을 구하고
    # 수직하는 방정식를 방향을 x,y 방향으로 해서 반환
    @abstractclassmethod
    def get_normal_vector(self, x:float, y:float)-> np.array:
        pass
    # 점에 대한 법선 벡트를 반환한다.
    @abstractclassmethod
    def get_reflect_closure(self, direct_vec, normal_vec, power):
        pass

class ICrashObserver(ICrash, IObserver, metaclass=ABCMeta):
    @abstractclassmethod
    def update(self, event:dict=None) -> None:
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
    def get_reflect_closure(self, direct_vec, normal_vec, power):
        pass
    # 충돌 했을 때 발생하는 이벤트를 받는다. 
    # crasher is closure
    @abstractclassmethod
    def crash(self, crashable:ICrashable):
        pass

class IMovableObserver(IMoveable, IObserver, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()
    @abstractclassmethod
    def get_xy(self)->list:
        pass
    @abstractclassmethod
    def update(self, event:dict=None) -> None:
        pass