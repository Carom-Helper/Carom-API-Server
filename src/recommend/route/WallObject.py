from action_cls import *
import numpy as np

def is_test_wallobject()->bool:
    return True and is_test()

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
        pos1_pos2 = [ pos2["x"]-pos1["x"], pos2["y"]-pos1["y"] ]
        
        self.wall_direct = np.array(pos1_pos2)
        self.wall_direct = self.wall_direct / np.linalg.norm(self.wall_direct)
        temp = self.wall_direct.tolist()
        temp = [-temp[1], -temp[0]]
        self.orth_vec = np.array(temp)
        test_print("init", "vector",self.wall_direct)
        test_print("init", "orth vector",self.orth_vec)
        test_print("init", "vec X orth_vec", self.wall_direct * self.orth_vec)
        
    def notify_filltered_observer(self, observer:IObserver)->None:
        pass

    def get_distance_from_point(self, x:float, y:float)-> float:
        p = self.pos1
        input = [x,y]
        input = np.array(input)
        

    def get_normal_vector(self, x:float, y:float)-> np.array:
        pass

    def get_reflect_closure(self):
        pass
    
    def get_xy(self)->list:
        pass
    def update(self, event:dict=None) -> None:
        pass
def test():
    print("top")
    wall_top = WallObject(
            pos1={"x":0.0, "y":0.0}, 
            pos2={"x":400.0, "y":0.0}
        )
    print("right")
    wall_right = WallObject(
            pos1={"x":400.0, "y":0.0}, 
            pos2={"x":400.0, "y":800.0}
        )
    print("bottom")
    wall_bottom = WallObject(
            pos1={"x":400.0, "y":800.0}, 
            pos2={"x":0.0, "y":800.0}
        )
    print("left")
    wall_left = WallObject(
            pos1={"x":0.0, "y":800.0}, 
            pos2={"x":0.0, "y":0.0}
        )
    
if __name__ == '__main__':
    test()