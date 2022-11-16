import argparse
import numpy as np
import math

from route_utills import is_test, print_args
from action_cls import *

def is_test_caromball()->bool:
    return True and is_test()

def test_print(s, s1="", s2="", s3="", s4="", s5="", end="\n"):
    if is_test_caromball():
        print("action cls exe : ", s, s1, s2, s3, s4, s5, end=end)


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
    

def test(
    cue_coord=(300,400), 
    tar1_coord=(350,450)
    ):
    cue = CaromBall()
    cue.start_param(clock = 12, tip = 1)
    cue.print_param()

    tar = CaromBall()
    tar.set_xy(*tar1_coord)

    cue.set_xy(*cue_coord)
    set_vec(cue, tar, 0)
    cue.set_mover(cue.move_by_time)
    print(cue.move(1))
    
def runner(args):
    print_args(vars(args))
    test(args.cue, args.tar1)
    #run(args.src, args.device)
    # detect(args.src, args.device)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cue', default=(300,400))
    parser.add_argument('--tar1', default=(350,450))
    # parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--display', action="store_true")
    args = parser.parse_args()
    runner(args) 