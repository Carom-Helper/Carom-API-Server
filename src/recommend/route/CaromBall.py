import argparse
import numpy as np
import math
import cv2

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
        self.moved = 0
        
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
        print(dist)
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
        dist = self.mover(t)
        if dist > 0:
            return self.notify_observers()
        else:
            return False

    def move_by_time(self, elapsed:float)->float:
        #decrease = [0.0121504, 0.0164196, 0.0207284, 0.0271995, 0.0338062, 0.0450838, 0.0572598, 0.0820890, 0.1394312, 1.0000000]
        decrease = [1.0000000, 0.1394312, 0.0820890, 0.0572598, 0.0450838, 0.0338062, 0.0271995, 0.0207284, 0.0164196, 0.0121504]

        x, y = self.xy[-1]["x"], self.xy[-1]["y"]
        new_x, new_y = x + self.vector["x"] * elapsed, y + self.vector["y"] * elapsed
        dist = ((new_x - x)**2 + (new_y - y)**2)**0.5
        self.moved += dist

        xy = {"x": new_x, "y": new_y, "elapsed": self.xy[-1]["elapsed"] + elapsed}
        self.add_xy(xy)


        if self.moved > 10:
            self.moved -= 10
            upspinmax = 3  * math.sin(math.pi * (90 / 180)) * 50 * self.radius
            upspinmin = 3  * math.sin(math.pi * (-60 / 180)) * 50 * self.radius
            next_power = self.power * (1-decrease[self.upspinrate-1])
            
            self.vector["x"] = self.vector["x"] * next_power / self.power
            self.vector["y"] = self.vector["y"] * next_power / self.power

            self.power = next_power
            #self.upspinrate = int((self.upspin - upspinmin) / (upspinmax-upspinmin) * 10)
            #print(self.power, 1-decrease[self.upspinrate-1])
        return dist
    
    def move_stay(self, elapsed:float)->dict:
        x, y = self.xy[-1]['x'], self.xy[-1]['y']

        xy = {"x": x, "y": y, "elapsed": self.xy[-1]["elapsed"] + elapsed}
        self.add_xy(xy)
        return 0
    
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
    tar1_coord=(100,750),
    tar2_coord=(300,300)
    ):
    cue = CaromBall()
    tar1 = CaromBall()
    tar2 = CaromBall()
    cue.start_param(clock = 12, tip = 1)
    cue.print_param()

    cue.set_xy(*cue_coord)
    tar1.set_xy(*tar1_coord)
    tar2.set_xy(*tar2_coord)

    cue.register_observer(tar1)
    cue.register_observer(tar2)

    set_vec(cue, tar1, 0)
    cue.set_mover(cue.move_by_time)
    tar1.set_mover(tar1.move_stay)
    tar2.set_mover(tar2.move_stay)

    elapsed = 1
    for _ in range(1000):
        c1 = cue.move(elapsed)
        c2 = tar1.move(elapsed)
        c3 = tar2.move(elapsed)
        if c1 or c2 or c3:
            break
        
    #if False:
    if True:
        show(cue, tar1, tar2)

def show(cue, tar1, tar2):
    img = np.zeros((800,400,3), np.uint8)
    for c in cue.get_xy():
        img = cv2.line(img, (int(c['x']), int(c['y'])), (int(c['x']), int(c['y'])), (255, 255, 255),1)
    for t in tar1.get_xy():
        img = cv2.line(img, (int(t['x']), int(t['y'])), (int(t['x']), int(t['y'])), (0, 0, 255),1)
    for t in tar2.get_xy():
        img = cv2.line(img, (int(t['x']), int(t['y'])), (int(t['x']), int(t['y'])), (0, 255, 0),1)
    cv2.imshow('simulate', img)
    cv2.waitKey()
    
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