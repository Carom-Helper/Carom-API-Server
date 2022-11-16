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
    def get_distance_from_point(self, x:float, y:float)-> float:
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
        #IMoveable.__init__()
        ICrashable.__init__(self)
        IObserver.__init__(self)
        ISubject.__init__(self)
        self.xy = []
        self.vector = {"x": 0, "y": 0}
        self.radius = 8.6
        
    def start_param(self, power = 50, clock = 12, tip = 0):
        self.power = power
        self.theta = clock % 12 * (-30) + 90
        self.tip = tip

        upspinmax = 3  * math.sin(math.pi * (90 / 180)) * 50 * self.radius
        upspinmin = 3  * math.sin(math.pi * (-60 / 180)) * 50 * self.radius
        self.upspin = math.sin(math.pi * (self.theta/180)) * tip * self.power * self.radius
        self.upspinrate = int((self.upspin - upspinmin) / (upspinmax-upspinmin) * 10)

        sidespinmax = 3 * math.cos(math.pi * (0 / 180)) * 50 * self.radius
        sidespinmin = 3 * math.cos(math.pi * (-180 / 180)) * 50 * self.radius
        self.sidespin = math.cos(math.pi * (self.theta/180)) * tip * self.power * self.radius
        self.sidespinrate = int((self.sidespin - sidespinmin) / (sidespinmax-sidespinmin) * 10)

    def print_param(self):
        print(f'theta: {self.theta}, tip: {self.tip}/3')
        print(f'upspin: {self.upspin:0.2f}, sidespin: {self.sidespin:0.2f}')
        print(f'upspinrate: {self.upspinrate}, sidespinrate: {self.sidespinrate}\n')
    
    def update(self, event:dict=None) -> bool:
        return self.get_distance_from_point(x=event['x'], y=event['y'])

    def notify_observers(self):
        self.crash_list = []
        for ob in self.observer_list:
            if ob.update(self.xy[-1]):
                self.crash_list.append(ob)
        return len(self.crash_list) > 0

    def get_distance_from_point(self, x:float, y:float)-> float:
        curr_pos = self.xy[-1]
        dist = ((curr_pos['x'] - x)**2 + (curr_pos['y'] - y)**2)**0.5
        return True if dist < self.radius * 2 else False

    def get_normal_vector(self, x:float, y:float)-> np.array:
        pass
    
    def crash(self, normal_vec:np.array):
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
        if xy is not None:
            self.add_xy(xy)
            return self.notify_observers()
        else:
            return False

    def move_by_time(self, elapsed:float)->float:
        x, y = self.xy[-1]["x"], self.xy[-1]["y"]
        new_x, new_y = x + self.vector["x"] * elapsed, y + self.vector["y"] * elapsed

        xy = {"x": new_x, "y": new_y, "elapsed": self.xy[-1]["elapsed"] + elapsed}
        return xy
    
    def move_stay(self, elapsed:float)->dict:
        x, y = self.xy[-1]['x'], self.xy[-1]['y']

        xy = {"x": x, "y": y, "elapsed": self.xy[-1]["elapsed"] + elapsed}
        return xy


def set_vec(cue:CaromBall, tar:CaromBall, thickness:float)->dict:
    cue_pos = cue.get_xy()[-1]
    tar_pos = tar.get_xy()[-1]

    cue_tar = {'x':(cue_pos['x'] - tar_pos['x']), 'y':(cue_pos['y'] - tar_pos['y'])}
    new_x = thickness/8 * cue.radius
    new_y = (cue.radius**2 - new_x**2)**0.5

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