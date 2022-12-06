from route_utills import angle, radian2degree
from action_cls import *
import numpy as np
from CaromBall import (
    CaromBall, 
    radius, upspinmax, upspinmin, upsinrange, sidespinmax, sidespinmin, sidespinrange
)
def is_test_wallobject()->bool:
    return False and is_test()

def test_print(s, s1="", s2="", s3="", s4="", s5="", end="\n"):
    if is_test_wallobject():
        print("wall object exe : ", s, s1, s2, s3, s4, s5, end=end)
        
class WallObject(IObserver, ICrashableSubject):
    elapse = 1.0
    def __init__(
            self,
            pos1={"x":0.0, "y":0.0}, 
            pos2={"x":800.0, "y":0.0},
            name="top"
        ) -> None:
        super().__init__()
        self.name=f'{name}'
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
            xy = observer.get_xy()
            x, y = xy
            # test_print("notify_filltered_observer",x,y)
            if self.orth_vec[0] == 0:
                x = self.pos1[0]
            else :
                y = self.pos1[1]
            distance = observer.get_distance_from_point(x,y)
            if (distance < self.elapse): # 충돌
        #       충돌을 전파한다.
                if isinstance(observer, ICrashAction):
                    event["crashable"] = self
        
        observer.update(event)

    def get_distance_from_point(self, x:float, y:float)-> float:
        # test_print("get_distance_from_point")
        p = np.array(self.pos1)
        vec = self.orth_vec
        input = [x,y]
        input = np.array(input)
        
        result = ((p - input)*vec).tolist()
        result = sum(result)
        return abs(result)
        

    def get_normal_vector(self, x:float, y:float)-> np.array:
        return self.orth_vec

    # closure는 방향벡터와 노멀벡터를 입력으로 받는다.
    def get_reflect_closure(self, direct_vec, normal_vec):
        # 정방향 좌우회전 구하기
        right_side = direct_vec[0] * (normal_vec.sum())
        # 입사각을 구한다.
        direct_vec = - direct_vec
        radian = angle(direct_vec, normal_vec)
        direct_normal_degree = radian2degree(radian)
        degree = 90.0 - direct_normal_degree

        #회전 방향
        outer = direct_vec[0] * normal_vec[1] - direct_vec[1] * normal_vec[0]
        dir = 1 if outer < 0 else -1

        bias_list = [
            [
                [-11.65999985,-3.420000029,-1.040000033,0.220000009,1.200000048,2.759999943,4.159999895,5.659999847,6.679999828,19.79999924],
                [5.619999886,7.199999809,7.900000095,8.600000381,9.199999809,9.620000267,10.43999977,11.19999981,11.89999962,23.29999924],
                [-5.599999905,-0.400000006,2.599999905,4.900000095,8.5,11.10000038,12.80000019,15.80000019,20.29999924,36.70000076],
                [-22.54999924,-13.10000038,-2.399999976,1.200000048,6.049999952,12.10000038,17.15000057,21.20000076,26.44999981,40.59999847]
            ],
            [
                [-1,2,3,4.199999809,5.199999809,5.699999809,6.599999905,7.199999809,8.100000381,19.79999924],
                [5.699999809,7.900000095,8.800000191,9.600000381,10.30000019,10.89999962,11.5,12.30000019,14.19999981,27.20000076],
                [9.399999619,12.5,13.39999962,14.5,15.69999981,17,18.29999924,19.89999962,23.20000076,37],
                [-7.439999771,-0.699999988,11.80000019,17.79999924,20.39999962,22.27999954,24.10000038,26.39999962,31.02000008,69.69999695]
            ],
            [
                [-1.600000024,1.680000043,2.5,2.960000038,3.75,4.699999809,5.900000095,6.940000057,8.110000324,14.60000038],
                [3.940000057,6.699999809,8.260000038,9.399999619,10.19999981,11,11.80000019,12.85999985,14.89999962,30.60000038],
                [8.75,12.10000038,13.60000038,15.10000038,16.39999962,17.60000038,18.79999924,20.20000076,21.94999981,35],
                [-1.419999981,9.459999847,17.60000038,20.5,21.79999924,22.70000076,23.89999962,25.10000038,28.31999931,53.29999924]
            ],
            [
                [-1.700000048,1.879999971,2.900000095,3.45999999,4.099999905,5.560000038,6.900000095,7.800000191,9.059999943,16.79999924],
                [2.099999905,5.199999809,7,8.5,9.399999619,10.60000038,12.19999981,14,16.89999962,35.5],
                [0.759999985,8.899999619,11.59000034,13.19999981,14.75,16.39999962,17.79999924,19.33999939,22.10000038,53.20000076],
                [-7.000000095,0.8,6.480000019,15.21999989,19.10000038,20.37999954,21.79999924,23.60000038,27.33999939,50.59999847]
            ],
            [
                [-0.869999981,0.0400000006,1,1.54000001,2.350000024,3.15999999,4.440000057,5.599999905,6.689999819,18.10000038],
                [-1.299999982,2.900000095,4.649999857,5.900000095,7.75,9.199999809,10,10.89999962,13.39999962,28.60000038],
                [-1.779999959,1.919999981,7.869999933,9.739999962,11.5,13.16000004,14.66999998,16,19.28999939,35.20000076],
                [-10.7,-4.960000038,-0.450000003,1.220000029,6.25,14.15999985,16.70000076,18.70000076,20.8699995,33.29999924]
            ]
        ]
        
        def simple_reflect_ball2wall(data:dict):
            bias_power = []
            bias_table = []
            for pi, pj in enumerate[3, 8, 15, 20, 100]:
                if data["power"] < pj:
                    bias_power = bias_list[pi]
                else:
                    break
            
            for di, dj in enumerate([22.5, 45, 60, 90]):
                if degree <= dj:
                    bias_table = bias_power[di]
                else:
                    break

            # reflect vec 구하기
            x , y = normal_vec.tolist()
            reflect_vec = direct_vec.copy()
            if x==0:
                reflect_vec[0] = -direct_vec[0]
            else:
                reflect_vec[1] = -direct_vec[1]
            
            # power, sidespin 힘
            power = data["power"]
            sidespin = data["sidespin"]
            
            # upspin = data["upspin"]
            # upspin_lv = int((upspin - upspinmin) / (upsinrange) * 10)
            
            #power 선택
            for speed_guide in bias_table["key"]:
                if float(power) < float(speed_guide):
                    test_print("speed_guide",power,speed_guide)
                    pick_key = speed_guide
            try:
                table = bias_table[str(pick_key)]
            except:
                print("error degree : ",degree, str(pick_key))
            # test_print("===============table=============\n", table)
            
            # sidespin 선택
            #   역회전 확인
            if sidespin * right_side > 0: #정회전
                sidespin_lv = int((sidespin - sidespinmin) / (sidespinrange) * 10)
                if sidespin_lv >= 10:
                    sidespin_lv = 9 
            else:#역회전
                sidespin_lv = int(1)
            
            # set new degree
            theta = table[sidespin_lv]
            radian = np.deg2rad(theta)
            
            # set new vector
            cos = np.cos(radian)
            sin = np.sin(radian)
            
            x = reflect_vec[0]
            y = reflect_vec[1]
            reflect_vec[0] = x * cos - y * sin
            reflect_vec[1] = x * sin + y * cos
            
            return (reflect_vec, data)

        def none_reflect(data : dict):
            reflect_vec = - direct_vec
            return reflect_vec, data

        if degree > 90:
            return none_reflect
        else:
            return simple_reflect_ball2wall
    
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
        
        print(result)
def test():
    test_get_distance_from_point()
    
if __name__ == '__main__':
    test()