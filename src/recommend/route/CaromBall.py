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

radius = 8.6
upspinmax = 3  * math.sin(math.pi * (90 / 180)) * 50 * radius
upspinmin = 3  * math.sin(math.pi * (-60 / 180)) * 50 * radius
upsinrange = upspinmax - upspinmin
sidespinmax = 3 * math.cos(math.pi * (0 / 180)) * 50 * radius
sidespinmin = 3 * math.cos(math.pi * (-180 / 180)) * 50 * radius
sidespinrange = sidespinmax - sidespinmin


class CaromBall(ICrashObserver, IMoveableSubject):
    def __init__(self, name="cue") -> None:
        IMoveableSubject.__init__(self)
        self.name=f'{name}'
        self.xy = []
        self.vector = {"x": 0, "y": 0}
        self.moved = 0
        self.colpoint = []
        self.last_crashable = None
        self.crash_list = []
        
    def start_param(self, power = 50, clock = 12, tip = 0):
        self.power = power
        self.theta = clock % 12 * (-30) + 90
        self.tip = tip

        self.upspin = math.sin(math.pi * (self.theta/180)) * tip * self.power * radius
        self.upspin_lv = int((self.upspin - upspinmin) / (upsinrange) * 10)

        self.sidespin = math.cos(math.pi * (self.theta/180)) * tip * self.power * radius
        self.sidespin_lv = int((self.sidespin - sidespinmin) / (sidespinrange) * 10)

    def print_param(self):
        print(f'theta: {self.theta}, tip: {self.tip}/3')
        print(f'upspin: {self.upspin:0.2f}, sidespin: {self.sidespin:0.2f}')
        print(f'upspin_lv: {self.upspin_lv}, sidespin_lv: {self.sidespin_lv}\n')
    
    def update(self, event:dict=None) -> bool:
        return self.get_distance_from_point(x=event['x'], y=event['y'])

    def notify_observers(self):
        self.crash_list = []
        for ob in self.observer_list:
            if ob.update():
                self.crash_list.append(ob)
        return len(self.crash_list) > 0

    def crash(self, crashable:ICrashable):
        if self.last_crashable is not crashable:
            v = np.array([self.vector['x'], self.vector['y']])
            #x, y = self.get_xy()
            closure = crashable.get_reflect_closure(v, crashable.get_normal_vector(*self.get_xy()))
            new_v, data = closure({"power": self.power, "upspin": self.upspin, "sidespin": self.sidespin})
            
            self.vector['x'] = new_v[0]
            self.vector['y'] = new_v[1]

            self.power = data['power']

            self.upspin = data['upspin']
            self.upspin_lv = int((self.upspin - upspinmin) / (upsinrange) * 10)

            self.sidespin = data['sidespin']
            self.sidespin_lv = int((self.sidespin - sidespinmin) / (sidespinrange) * 10)

            self.colpoint.append([int(self.xy[-1]['x']), int(self.xy[-1]['y'])])
            self.last_crashable = crashable
            self.crash_list.append(self.last_crashable.name)

    def get_distance_from_point(self, x:float, y:float)-> float:
        curr_pos = self.xy[-1]
        dist = ((curr_pos['x'] - x)**2 + (curr_pos['y'] - y)**2)**0.5 - radius
        return dist

    def notify_filltered_observer(self, observer:IObserver)->None:
        pass

    def get_normal_vector(self, x:float, y:float)-> np.array:
        pass
    
    def get_reflect_closure(self):
        pass

    def get_xy(self)->list:
        return self.xy[-1]['x'], self.xy[-1]['y']

    def set_xy(self, x:float, y:float):
        temp = {"x": x, "y": y, "elapsed": 0}
        self.xy.append(temp)

    def add_xy(self, xy:dict):
        self.xy.append(xy)

    def move(self, t:float)->float:
        self.crash_list = []
        dist, next_t = self.mover(t)
        if dist > 0:
            self.notify_observers()
            cue_hit = self.crash_list
            return cue_hit, dist, next_t
        else:
            return None, 0, t

    def move_by_time(self, elapsed:float)->float:
        decrease = [
            [1.0000000, 0.1394312, 0.0820890, 0.0572598, 0.0450838, 0.0338062, 0.0271995, 0.0207284, 0.0164196, 0.0121504],
            [1.0000000, 0.0503756, 0.0293241, 0.0194037, 0.0140439, 0.0105088, 0.0083705, 0.0067304, 0.0054193, 0.0039934],
            [0.8346197, 0.0249414, 0.0133947, 0.0086298, 0.0061542, 0.0047085, 0.0036647, 0.0029340, 0.0022524, 0.0016758],
            [0.6223567, 0.0087800, 0.0044792, 0.0028660, 0.0020876, 0.0016103, 0.0012504, 0.0009685, 0.0007472, 0.0006030],
            [0.4842837, 0.0037113, 0.0021989, 0.0015158, 0.0012120, 0.0010040, 0.0008037, 0.0006748, 0.0005585, 0.0004263]
        ]

        x, y = self.xy[-1]["x"], self.xy[-1]["y"]
        new_x, new_y = x + self.vector["x"] * elapsed, y + self.vector["y"] * elapsed
        dist = ((new_x - x)**2 + (new_y - y)**2)**0.5
        self.moved += dist

        xy = {"x": new_x, "y": new_y, "elapsed": self.xy[-1]["elapsed"] + elapsed}
        self.add_xy(xy)

        next_elapsed = (3/5) / (dist/elapsed)

        if self.moved > 1:
            self.moved -= 1
            upspinmax = 3  * math.sin(math.pi * (90 / 180)) * 50 * radius
            upspinmin = 3  * math.sin(math.pi * (-60 / 180)) * 50 * radius
            for i, j in enumerate([1, 2.5, 4.5, 7, 100]):
                if self.power < j:
                    decreaserate = (1-decrease[i][self.upspin_lv if self.upspin_lv < 10 else 9])
                    break
            next_power = self.power * decreaserate
            
            reduce = next_power / self.power
            
            self.vector["x"] = self.vector["x"] * reduce
            self.vector["y"] = self.vector["y"] * reduce

            self.power = next_power
            next_elapsed = (3/5) / (dist/elapsed*reduce)

            #self.upspin_lv = int((self.upspin - upspinmin) / (upspinmax-upspinmin) * 10)
            #print(self.power, 1-decrease[self.upspin_lv-1])
        return dist/elapsed, next_elapsed
    
    def move_stay(self, elapsed:float)->dict:
        x, y = self.xy[-1]['x'], self.xy[-1]['y']

        xy = {"x": x, "y": y, "elapsed": self.xy[-1]["elapsed"] + elapsed}
        self.add_xy(xy)
        return 0, elapsed
    
def set_vec(cue:CaromBall, tar:CaromBall, thickness:float)->dict:
    cue_x, cue_y = cue.get_xy()
    tar_x, tar_y = tar.get_xy()

    cue_tar = {'x':(cue_x - tar_x), 'y':(cue_y - tar_y)}
    new_x = thickness/8 * radius
    new_y = (radius**2 - new_x**2)**0.5

    new_x *= 1.5
    new_y *= 1.5

    new_t = {'x': new_x, 'y': new_y}

    cue_tar_l = (cue_tar['x']**2+cue_tar['y']**2)**0.5
    cos = cue_tar['y'] / cue_tar_l
    sin = -cue_tar['x'] / cue_tar_l

    new_t['x'] = new_x * cos - new_y * sin + tar_x
    new_t['y'] = new_x * sin + new_y * cos + tar_y

    vector = {"x": new_t["x"] - cue_x, "y": new_t["y"] - cue_y}

    length = (vector["x"]**2 + vector["y"]**2)**0.5
    vector["x"] *= 3/5 / length * cue.power / 50
    vector["y"] *= 3/5 / length * cue.power / 50
    
    cue.vector["x"] = vector["x"]
    cue.vector["y"] = vector["y"]
    

def test(
    cue_coord=(300,400), 
    tar1_coord=(100,750),
    tar2_coord=(300,300),
    power = 50,
    clock = 12,
    tip = 1,
    thick = 0
    ):
    cue = CaromBall()
    tar1 = CaromBall()
    tar2 = CaromBall()

    cue.start_param(power = power, clock = clock, tip = tip)
    cue.print_param()

    cue.set_xy(*cue_coord)
    tar1.set_xy(*tar1_coord)
    tar2.set_xy(*tar2_coord)

    cue.register_observer(tar1)
    cue.register_observer(tar2)

    set_vec(cue, tar1, thick)
    cue.set_mover(cue.move_by_time)
    tar1.set_mover(tar1.move_stay)
    tar2.set_mover(tar2.move_stay)

    elapsed = 1
    i=0
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
    clist = cue.xy
    img = cv2.line(img, (int(clist[0]['x']), int(clist[0]['y'])), (int(clist[0]['x']), int(clist[0]['y'])), (255, 255, 255), 3)
    t1list = tar1.xy
    img = cv2.line(img, (int(t1list[0]['x']), int(t1list[0]['y'])), (int(t1list[0]['x']), int(t1list[0]['y'])), (0, 0, 255), 3)
    t2list = tar2.xy
    img = cv2.line(img, (int(t2list[0]['x']), int(t2list[0]['y'])), (int(t2list[0]['x']), int(t2list[0]['y'])), (0, 255, 0), 3)

    for c in clist:
        img = cv2.line(img, (int(c['x']), int(c['y'])), (int(c['x']), int(c['y'])), (255, 255, 255), 1)
    for t in t1list:
        img = cv2.line(img, (int(t['x']), int(t['y'])), (int(t['x']), int(t['y'])), (0, 0, 255), 1)
    for t in t2list:
        img = cv2.line(img, (int(t['x']), int(t['y'])), (int(t['x']), int(t['y'])), (0, 255, 0), 1)
    cv2.imshow('simulate', img)
    cv2.waitKey()
    
def runner(args):
    print_args(vars(args))
    
    for c in range(12):
        for t in range(1,4):
            for p in range(10, 60, 10):
                for th in range(-7, 8):
                    # c = 5
                    # p = 50
                    # th = 0
                    print(c, t, p, th)
                    #test(args.cue, args.tar1, power=p, clock=c, tip=t, thick=th)
                    test(power=p, clock=c, tip=t, thick=th)
    
    #test(power = 50, clock = 5, tip = 3, thick = 0)
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