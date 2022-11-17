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
            pos2={"x":800.0, "y":0.0},
            name="top"
        ) -> None:
        super().__init__()
        self.name=name
        self.pos1=[pos1["x"], pos1["y"]]
        self.pos2=[pos2["x"], pos2["y"]]
        pos1_pos2 = [ pos2["x"]-pos1["x"], pos2["y"]-pos1["y"] ]
        
        self.wall_direct = np.array(pos1_pos2)
        self.wall_direct = self.wall_direct / np.linalg.norm(self.wall_direct)
        temp = self.wall_direct.tolist()
        temp = [-temp[1], -temp[0]]
        self.orth_vec = np.array(temp)
        # test_print("init", "vector",self.wall_direct)
        # test_print("init", "orth vector",self.orth_vec)
        # test_print("init", "vec X orth_vec", self.wall_direct * self.orth_vec)
    
    # 특정 옵져버에 대해서 해야할 행동을 정의한다.
    def notify_filltered_observer(self, observer:IObserver)->None:
        # 들어오는 옵져버는 Crashable 옵져버가 들어온다.
        # 움직임을 확인하고
        if observer is IMovableObserver:
        #   1. ICrashObserver가 들어오면,
            if (observer is ICrashable and
                observer is ICrashAction):
        #       충돌을 확인하고,
                xy = observer.get_xy()
                if self.check_crash()
        #       충돌을 전파한다.
        

    def get_distance_from_point(self, x:float, y:float)-> float:
        test_print("get_distance_from_point")
        p = np.array(self.pos1)
        vec = self.orth_vec
        input = [x,y]
        input = np.array(input)
        
        test_print(p, input)
        result = ((p - input)*vec).tolist()
        test_print("list", result)
        result = sum(result)
        test_print("sum", result)
        return abs(result)
        

    def get_normal_vector(self, x:float, y:float)-> np.array:
        return self.orth_vec

    def get_reflect_closure(self):
        pass
    
    def get_xy(self)->list:
        return self.pos1
    
    def update(self, event:dict=None) -> None:
        # crash 이벤트를 전파해야한다.
        self.notify_observers()
    
    
def test_get_distance_from_point():
    from random import randint
    wall_list = list()
    print("top")
    wall_list.append(WallObject(
            pos1={"x":0.0, "y":0.0}, 
            pos2={"x":400.0, "y":0.0},
            name="top"
        ))
    print("right")
    wall_list.append(WallObject(
            pos1={"x":400.0, "y":0.0}, 
            pos2={"x":400.0, "y":800.0},
            name="right"
        ))
    print("bottom")
    wall_list.append(WallObject(
            pos1={"x":400.0, "y":800.0}, 
            pos2={"x":0.0, "y":800.0},
            name="bottom"
        ))
    print("left")
    wall_list.append(WallObject(
            pos1={"x":0.0, "y":800.0}, 
            pos2={"x":0.0, "y":0.0},
            name="left"
        ))
    point = (randint(8,392), randint(8,792))
    print("point", point)
    for wall in wall_list:
        result = dict()
        result["name"] = wall.name
        result["orth"] = wall.orth_vec
        result["distance"] = wall.get_distance_from_point(*point)
        result["xy"] = wall.get_xy()
        
        print(result)
def test():
    test_get_distance_from_point()
    
if __name__ == '__main__':
    test()