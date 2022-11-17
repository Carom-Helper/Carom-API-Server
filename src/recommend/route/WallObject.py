from action_cls import *
import numpy as np

def is_test_wallobject()->bool:
    return False and is_test()

def test_print(s, s1="", s2="", s3="", s4="", s5="", end="\n"):
    if is_test_wallobject():
        print("wall object exe : ", s, s1, s2, s3, s4, s5, end=end)
        
class WallObject(IMovableObserver, ICrashableSubject):
    def __init__(
            self,
            pos1={"x":0.0, "y":0.0}, 
            pos2={"x":800.0, "y":0.0}
        ) -> None:
        super().__init__()
        self.pos1=pos1
        self.pos2=pos2
        self.wall_direct = np.vectorize()
        
    
    def notify_filltered_observer(self, observer:IObserver)->None:
        pass
    
    def get_tangent_line(self, x:float, y:float)-> float:
        
    
    def get_distance_from_point(self, x:float, y:float)-> float:
        pass
    
    def get_normal_vector(self, x:float, y:float)-> np.array:
        pass

    def get_reflect_closure(self):
        pass
    