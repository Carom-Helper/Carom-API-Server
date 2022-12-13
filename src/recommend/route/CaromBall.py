import argparse
import numpy as np
import math
import cv2

# import route
from route_utills import is_test, print_args
from action_cls import *
def is_test_caromball()->bool:
    return False and is_test()

def test_print(s, s1="", s2="", s3="", s4="", s5="", end="\n"):
    if is_test_caromball():
        print("Carom Ball cls exe : ", s, s1, s2, s3, s4, s5, end=end)

radius = 8.6
upspinmax = 3  * math.sin(math.pi * (90 / 180)) * 50 * radius
upspinmin = 3  * math.sin(math.pi * (-60 / 180)) * 50 * radius
upsinrange = upspinmax - upspinmin
sidespinmax = 3 * math.cos(math.pi * (0 / 180)) * 50 * radius
sidespinmin = 3 * math.cos(math.pi * (-180 / 180)) * 50 * radius
sidespinrange = sidespinmax - sidespinmin


class CaromBall(IObserver, ICrash, IMoveable, IFitteringSubject):
    elapse = 0
    def __init__(self, name="cue") -> None:
        IFitteringSubject.__init__(self)
        self.name=f'{name}'
        self.xy = []
        self.vector = {"x": 0, "y": 0}
        self.moved = 0
        self.colpoint = []
        self.last_crashable = None
        self.crash_list = []
        self.thick = 0
        self.power=0
        self.upspin=0
        self.sidespin=0
        self.new_v = [0, 0]
        self.new_power = 0
        self.data = None
        self.wall_v = None
    def __str__(self) -> str:
        return f"[{self.name}]"+super().__str__()
    def start_param(self, power = 50, clock = 12, tip = 0):
        self.power = power
        self.theta = clock % 12 * (-30) + 90
        self.tip = tip

        self.upspin = math.sin(math.pi * (self.theta/180)) * tip * self.power * radius
        self.upspin_lv = int((self.upspin - upspinmin) / (upsinrange) * 10)
        if self.upspin_lv == 10:
            self.upspin_lv = 9

        self.sidespin = math.cos(math.pi * (self.theta/180)) * tip * self.power * radius
        self.sidespin_lv = int((self.sidespin - sidespinmin) / (sidespinrange) * 10)
        if self.sidespin_lv == 10:
            self.sidespin_lv = 9
        

    def print_param(self):
        print(f'theta: {self.theta}, tip: {self.tip}/3, thick: {self.thick}')
        print(f'upspin: {self.upspin:0.2f}, sidespin: {self.sidespin:0.2f}')
        print(f'upspin_lv: {self.upspin_lv}, sidespin_lv: {self.sidespin_lv}\n')
    
    def update(self, event:dict=None) -> None:
        # test_print("update move", self.get_xy())
        # if "elapsed" in event: # 움직인다.
        #     self.move(event["elapsed"]) # elapsed = t(float)
        # if "crashable" in event: # crash 를 일으킨다
        #     self.crash(event["crashable"])        
        """
        if self.new_v is not None:
            self.vector['x'] = self.new_v[0] * (0.6) * self.data["power"] / 50
            self.vector['y'] = self.new_v[1] * (0.6) * self.data["power"] / 50
            self.new_v = None
        """
        if self.wall_v is not None:
            self.vector['x'] = self.wall_v[0] * (0.6) * self.data["power"] / 50
            self.vector['y'] = self.wall_v[1] * (0.6) * self.data["power"] / 50
            self.wall_v = None

            self.power = self.data['power']

            self.upspin = self.data['upspin']
            self.upspin_lv = int((self.upspin - upspinmin) / (upsinrange) * 10)
            if self.upspin_lv == 10:
                self.upspin_lv = 9

            self.sidespin = self.data['sidespin']
            self.sidespin_lv = int((self.sidespin - sidespinmin) / (sidespinrange) * 10)
            if self.sidespin_lv == 10:
                self.sidespin_lv = 9
            self.data = None

        else:
            if self.data is not None:
                self.vector['x'] += self.data["vector"][0] + self.new_v[0]
                self.vector['y'] += self.data["vector"][1] + self.new_v[1]
                x,y = self.rotate_vector(self.data["rotate"], self.vector['x'], self.vector['y'])
                self.vector['x'], self.vector['y'] = x, y

                self.data["vector"] = [0, 0]
                self.new_v = [0, 0]

                self.power = self.data['power'] + self.new_power
                self.new_power = 0

                self.upspin = self.data['upspin']
                self.upspin_lv = int((self.upspin - upspinmin) / (upsinrange) * 10)
                if self.upspin_lv == 10:
                    self.upspin_lv = 9

                self.sidespin = self.data['sidespin']
                self.sidespin_lv = int((self.sidespin - sidespinmin) / (sidespinrange) * 10)
                if self.sidespin_lv == 10:
                    self.sidespin_lv = 9
                self.data = None
        dist = (self.vector['x']**2 + self.vector['y']**2)**0.5
        next_elapsed = 10000
        if dist > 0:
            next_elapsed = 0.6 / dist
        return next_elapsed

    def get_distance_from_point(self, x:float, y:float)-> float:
        curr_pos = self.xy[-1]
        dist = ((curr_pos['x'] - x)**2 + (curr_pos['y'] - y)**2)**0.5 - radius
        return dist
    
    def move(self, elapsed:float)->float:
        self.crash_list.clear()
        dist, next_elapsed = self.mover(elapsed)
        return dist, next_elapsed
        
    def notify_filltered_observer(self, observer:IObserver)->None:
        if not isinstance(observer, IObserver):
            print("Not in IObserver")
            return
        event = dict()
        # 들어오는 옵져버는 Crashable 옵져버가 들어온다.
        # 움직임을 확인하고
        if isinstance(observer,ICrashable):
        #   1. ICrashObserver가 들어오면,
            #if observer is ICrashable:
        #       충돌을 확인하고,
            distance = observer.get_distance_from_point(*self.get_xy())
            if (distance - radius < self.elapse): # 충돌
        #       충돌을 전파한다.
        
                self.crash(observer)
                test_print("notify_filltered_observer", f"====== {str(observer)} ======")
                event["crashable"] = self
            

    def get_normal_vector(self, x:float, y:float)-> np.array:
        x0, y0 = self.get_xy()
        vec = np.array([ x-x0, y-y0])
        vec = vec / np.linalg.norm(vec)
        test_print("nomal", vec)
        return vec
    
    def get_reflect_closure(self, direct_vec, normal_vec):
        crash_degree_table = [
            0,
            29,
            41,
            51,
            60,
            68,
            75,
            83
        ]
        direct_vec = - direct_vec
        thick = self.thick
        cue_degree = crash_degree_table[abs(thick)]
        bias_table = []
        if cue_degree < 20:
            bias_table = [-6.794786453, 5.675224304, 17.71131229, 20.55949783, 22.92873287,
            30.92001724, 35.10299683, 38.13070297, 39.05358505, 41.48766327]
        elif cue_degree < 26:
            bias_table = [ 1.101851463, 6.09233284, 9.714725685, 14.03004494, 15.45781517,
                18.12297783, 20.83201427, 22.78761215,  25.74553318, 28.90215111]
        elif cue_degree < 35:
            bias_table = [2.255041504, 7.60682106, 9.58551178, 11.45974808, 13.11029053,
                14.67814026, 17.26420975, 18.74237976, 21.18320007, 28.51823425]
        elif cue_degree < 44:
            bias_table = [3.138310242, 6.402128601, 8.470178604, 9.664089966, 11.30900574,
                12.51847, 14.27773552, 15.99832077, 18.13723202, 28.72849083]
        elif cue_degree < 55:
            bias_table = [-0.430860901, 4.952252579, 6.974358368, 8.682728577, 10.11644363,
                11.26008301, 12.23267441, 13.2471981, 14.67285652, 26.29415512]
        elif cue_degree <= 90.1:
            bias_table = [-1.001932144, 3.712745857, 5.213981628, 6.175741959, 7.22426033,
                8.593474197, 9.567179108, 10.51088638, 11.51598625, 28.40719414]
        else:
            raise TypeError("get_reflect_closure+Ball")
        
        split_table = {
            "key":[20, 21, 25,30,35,40,45,50,55,60,65,70,75],
            "20":80, "21":70, "25":60, "30":50, "35":40, "40":33, "45":30, "50":25,
            "55":20, "60":14, "65":12, "70":10, "75":10,}
        bias_power = 0
        for value in split_table["key"]:
            if cue_degree < int(value):
                bias_power = split_table[str(value)]
        bias_power = 50
        
        
        def simple_reflect_ball2ball(data:dict):
            #self = tar / data = cue
            # 방향벡터와 노멀벡터를 통해 반사벡터를 구함
            normal_direct_vec = direct_vec
            reflect_vec = 2*(np.dot(direct_vec, normal_vec))*normal_vec - direct_vec
            test_print("simple_reflect_ball2ball", data, direct_vec, normal_vec, reflect_vec)
            # 편이 각 구하기
            power = data["power"]
            upspin = data["upspin"]
            upspin_lv = int((upspin - upspinmin) / (upsinrange) * 10)
            if upspin_lv == 10:
                upspin_lv = 9

            reflect_vec = [normal_vec[1], -normal_vec[0]]
            
            bias_degree = bias_table[upspin_lv]
            radian = np.deg2rad(bias_degree)
            dot = np.dot(direct_vec, reflect_vec)
            if dot > 0:
                reflect_vec = [-reflect_vec[0], -reflect_vec[1]]
            
            # # set new vector
            reflect_vec = self.rotate_vector(bias_degree,-reflect_vec[0], -reflect_vec[1])
            # cos = np.cos(radian)
            # sin = np.sin(radian)
            
            # x = reflect_vec[0]
            # y = reflect_vec[1]
            # reflect_vec[0] = x * cos - y * sin
            # reflect_vec[1] = x * sin + y * cos
            
            # set power
            data["power"] = power * (100 -bias_power) * 0.01
            test_print("simple_reflect_ball2ball", data)
            
            return (reflect_vec, data)
        
        def complex_reflect_ball2ball(data:dict):
            # 0 <= separ <= 1
            dot = np.dot(direct_vec, normal_vec)
            d1 = (direct_vec[0]**2 + direct_vec[1]**2)**0.5
            d2 = (normal_vec[0]**2 + normal_vec[1]**2)**0.5
            cos = dot / d1 / d2
            rad = math.acos(cos)
            separ = rad * 2 / math.pi

            outer = direct_vec[0] * normal_vec[1] - direct_vec[1] * normal_vec[0]
            dir = 1 if outer < 0 else -1

            #tar to cue
            reflect_vec = normal_vec
            reflect_vec[0] = reflect_vec[0] * (1-separ) * (0.6) * data["power"] / 50
            reflect_vec[1] = reflect_vec[1] * (1-separ) * (0.6) * data["power"] / 50

            data["vector"][0] += reflect_vec[0]
            data["vector"][1] += reflect_vec[1]

            upspin = data["upspin"]
            upspin_lv = int((upspin - upspinmin) / (upsinrange) * 10)
            if upspin_lv == 10:
                upspin_lv = 9
            bias_degree = bias_table[9-upspin_lv]
            data["rotate"] = bias_degree * dir

            #cue to tar
            crash_vec = -normal_vec
            crash_vec[0] = crash_vec[0] * separ * (0.6) * data["power"] / 50
            crash_vec[1] = crash_vec[1] * separ * (0.6) * data["power"] / 50

            self.new_v = crash_vec

            prev_power = data["power"]
            data["power"] *= separ
            self.new_power = prev_power - data["power"]


            return None, data

        def first_crash_ball2ball(data:dict):
            power = self.power - self.data['power']
            data["power"] = power
            data['upspin'] = (upspinmax + upspinmin)/2
            data['rotate'] = 0


            return None, data
            #return reflect_vec, data

        if direct_vec[0] == 0 and direct_vec[1] == 0:
            return first_crash_ball2ball
        else:
            #return simple_reflect_ball2ball
            return complex_reflect_ball2ball
        

    def crash(self, crashable:ICrashable):
        if self.last_crashable is not crashable:
            test_print("Cue crachable : ", str(crashable), self.power)
            v = np.array([self.vector['x'], self.vector['y']])
            #x, y = self.get_xy()
            closure = crashable.get_reflect_closure(v, crashable.get_normal_vector(*self.get_xy()))
            #self.new_v, self.data = closure({"power": self.power, "upspin": self.upspin, "sidespin": self.sidespin, "vector": [0, 0]})
            self.wall_v, self.data = closure({"power": self.power, "upspin": self.upspin, "sidespin": self.sidespin, "vector": [0, 0]})

            self.colpoint.append([int(self.xy[-1]['x']), int(self.xy[-1]['y'])])
            self.last_crashable = crashable
            self.crash_list.append(crashable.name)

            if self.wall_v is not None:
                if self.wall_v[0] == v[0] and self.wall_v[1] == v[1]:
                    self.crash_list.remove(crashable.name)
            if isinstance(crashable, IMoveable):
                crashable.set_mover(crashable.move_by_time)
                self.remove_observer(crashable)
        
    def set_colpoint(self, x:float, y:float):
        if len(self.colpoint) > 0:
            px, py = self.colpoint[-1]
            if px != int(x) or py != int(y):
                self.colpoint.append([int(x), int(y)])
        else:
            self.colpoint.append([int(x), int(y)])
        
    def get_xy(self)->list:
        return self.xy[-1]['x'], self.xy[-1]['y']

    def set_xy(self, x:float, y:float):
        temp = {"x": x, "y": y, "elapsed": 0}
        self.set_colpoint(x,y)
        self.xy.append(temp)

    def add_xy(self, xy:dict):
        self.xy.append(xy)

    

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

        #next_elapsed = (3/5) / (dist/elapsed)
        if dist != 0:
            next_elapsed = 0.6 * elapsed / dist
        else:
            next_elapsed = elapsed

        if self.moved > 2:
            self.moved -= 2
            upspinmax = 3  * math.sin(math.pi * (90 / 180)) * 50 * radius
            upspinmin = 3  * math.sin(math.pi * (-60 / 180)) * 50 * radius
            decreaserate=1
            for i, j in enumerate([1, 2.5, 4.5, 7, 100]):
                if self.power < j:
                    decreaserate = (1-decrease[i][self.upspin_lv])
                    break
            next_power = self.power * decreaserate
            if next_power < 0.0005:
                next_power = 0
                self.set_mover(self.move_stay)
            #print(self.name, self.moved, next_power, self.power, self.vector)
            reduce = (next_power / self.power) if next_power > 0 else 1
            #print(next_power, reduce)
            
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
        return 0, 100000
    
    def rotate_vector(self, theta, x, y):
        radian = np.deg2rad(theta)
        cos = np.cos(radian)
        sin = np.sin(radian)

        rx = x * cos - y * sin
        ry = x * sin + y * cos

        return [rx, ry]
    
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
    cue.thick = thickness
    

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
    cv2.waitKey(1000)
    
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