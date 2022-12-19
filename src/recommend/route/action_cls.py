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
        #IFitteringNotifier.__init__()
        ISubject.__init__(self)
    @abstractclassmethod
    def notify_filltered_observer(self, observer:IObserver)->None:
        pass
    
    def notify_observers(self)->None:
        for observer in self.observer_list:
            self.notify_filltered_observer(observer)
        return observer
    

class IMoveable(metaclass=ABCMeta):
    mover = None
    
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


# 공이 충돌하면, 운동이 변경된다.
# 하지만 해당 운동을 바로 적용시키는 것보다는
# 다음 운동상태로 초기화 시키고,
# 의도적으로 호출하여 적용시키는 것이 더 상황을 컨트롤하기 좋았다.
# 해당 행동을 통일하기 위해서 의도적으로 만든 인터페이스 이다.
class LAZY_ACTION_SETTER(metaclass=ABCMeta):
    
    @abstractclassmethod
    def set_action(vector, power:float, spin:dict)->None:
        pass
    
    @abstractclassmethod
    def apply_action()->None:
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
    def get_reflect_closure(self, direct_vec, normal_vec):
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


class ICrashableSubject(ICrashable, IFitteringSubject, metaclass=ABCMeta):
    def __init__(self) -> None:
        ICrashable.__init__(self)
        IFitteringSubject.__init__(self)

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