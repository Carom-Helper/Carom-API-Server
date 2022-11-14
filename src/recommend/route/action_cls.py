from abc import *

from route_utills import is_test

def is_test_action_cls()->bool:
    return False and is_test()

def test_print(s, s1="", s2="", s3="", s4="", s5="", end="\n"):
    if is_test_action_cls():
        print("action cls exe : ", s, s1, s2, s3, s4, s5, end=end)

class ICrashObserver(metaclass=ABCMeta):
    @abstractclassmethod
    def crash(reflect_vec) -> None:
        pass

class ISubject(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.observer_list=list()    
    
    @abstractclassmethod
    def register_observer(observer:ICrashObserver):
        pass
    
    @abstractclassmethod
    def remove_observer(observer:ICrashObserver):
        pass
    
    @abstractclassmethod
    def notify_observers():
        pass
    
class IMoveable(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.distance_vector = 