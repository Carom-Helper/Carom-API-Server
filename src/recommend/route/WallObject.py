from route_utills import angle, radian2degree, rotate_vector
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
        
class WallObject(IObserver, ICrashable):
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
        #   서로 크로스 해서 곱하면, 정방향 회전과 관련된 일정한 패턴이 나온다.
        #   좌우스핀이 +일때 정방향인 경우는
        #   노멀 벡터의 영향력이 Y:+ X:- 로 나온다.
        #   이 규칙을 사용해서 정방향 좌우스핀을 구한다.
        #   해당 값에서 양수가 나오면 sidespin이 양수가 나와야 정방향 스핀이고,
        #   해당 값에서 음수가 나오면 sidespin이 음수가 나와야 정방향 스핀이다.
        right_side = (direct_vec[0] * normal_vec[1] + direct_vec[1] * normal_vec[0])
        if normal_vec[0] != 0:#X방향 영향력 일때
            right_side = right_side * -1
        
        # 입사각을 구한다.
        reverse_direct_vec = - direct_vec
        radian = angle(reverse_direct_vec, normal_vec)
        direct_normal_degree = radian2degree(radian)
        degree = 90.0 - direct_normal_degree

        # 진행 방향을 그대로 진행해야하는 경우 반환하는 벡터
        if degree > 90 or np.isnan(degree):
            #   노멀벡터와 방향벡터의 각이 90도가 넘었을 때는 
            #   공이 벽과 너무 가까워서 샷이 되자마자 충돌 판정이 났을 경우이다.
            #   이때 튕기면 매우 곤란하다.
            def none_reflect(data : dict):
                reflect_vec = direct_vec
                return reflect_vec, data
            return none_reflect

        bias_table = {"key":[0.5, 4, 7, 11, 1000]}
        #if power < 0.5:
        bias_level = str(bias_table["key"][0]) #"0.5"
        if degree < 30:
            bias_table[bias_level] = [8.839999962, 10.08000002, 11.32000008, 12.56000013,
            13.80000019, 15.5, 17.19999981, 18.89999962, 20.59999943, 22.29999924]
        elif degree < 45:
            bias_table[bias_level] = [0, 3.380000019, 4.300000095, 5.240000057, 6.599999905, 
                6.599999905, 7.259999847, 8.460000038, 10.26000023, 12.89999962]
        elif degree < 60: 
            bias_table[bias_level] = [-23.77999954, -19.70000076, -10.16000042, -3.080000114,
            -2.099999905, 1.7, 7.259999943, 9.719999886, 13.72000027, 23]
        elif degree <= 90.2:
            bias_table[bias_level] = [-18.26000061, -17.35999947, -12.54000006, -3.139999986,
                0.550000012, 4.420000076, 5.429999876, 24.99999924, 26.66000042, 31.39999962]
        else:
            raise ValueError(f"{degree},{bias_level})Over 90 degree.+get_reflect_closure")
        #elif power < 4:
        bias_level = str(bias_table["key"][1]) #"4"
        if degree < 10:
            bias_table[bias_level] = [
                -70.51999855,
                -3.460000038,
                -2.5,
                -1.420000029,
                -0.600000024,
                0.0200000047683713,
                0.75999999,
                1.95999999,
                2.759999943,
                4.900000095
            ]
        elif degree < 55:
            bias_table[bias_level] = [
                0,
                7.159999847,
                8.399999619,
                9.320000076,
                10.19999981,
                11.10000038,
                11.89999962,
                13.10000038,
                15.30000019,
                36.70000076
            ]
        elif degree < 85:
            bias_table[bias_level] = [
                -6.950000048,
                -1.100000024,
                2.850000024,
                8.300000191,
                12.89999962,
                16.20000076,
                18.39999962,
                20.60000038,
                26.20000076,
                40.59999847,
            ]
        elif degree <= 90.2:
            bias_table[bias_level] = [
                -29.56000061,
                -26.20000076,
                -18.50000076,
                -2.099999905,
                -0.250000007,
                1.979999971,
                8.489999866,
                20.61999969,
                25.72000065,
                36.20000076
            ]
        else:
            raise ValueError(f"{degree},{bias_level})Over 90 degree.+get_reflect_closure")
        #elif power < 7:
        bias_level = str(bias_table["key"][2]) #"7"
        if degree < 30:
            bias_table[bias_level] = [
                2,
                3.360000038,
                4.900000095,
                5.800000191,
                6.599999905,
                7.400000095,
                7.709999847,
                8.199999809,
                9.940000057,
                27.20000076,
            ]
        elif degree < 55:
            bias_table[bias_level] = [
                0,
                9.300000191,
                10.47999992,
                11.30000019,
                12.19999981,
                13,
                13.60000038,
                14.69999981,
                17.10000038,
                29.20000076
            ]
        elif degree < 70:
            bias_table[bias_level] = [
                12.42999973,
                15.69999981,
                17,
                17.81999931,
                18.70000076,
                19.47999992,
                20.40999966,
                22,
                26.89999962,
                37,
            ]
        elif degree <= 90.1:
            bias_table[bias_level] = [
                -10.69999981,
                -0.899999976,
                10.20000029,
                16.79999924,
                20.84999943,
                23.10000038,
                24.75,
                28.20000076,
                31.65000057,
                56
            ]
        else:
            raise ValueError(f"{degree},{bias_level})Over 90 degree.+get_reflect_closure")
        #elif power < 11:
        bias_level = str(bias_table["key"][3]) #"11"
        if degree < 30:
            bias_table[bias_level] = [
                -0.499999994,
                2.099999905,
                3,
                4.260000038,
                5.099999905,
                6.699999809,
                7.5,
                9.5,
                11.10000038,
                16.39999962
            ]
        elif degree < 45:
            bias_table[bias_level] = [
                0,
                7.639999866,
                9,
                9.899999619,
                10.60000038,
                11.5,
                12.10000038,
                13.10000038,
                15.60000038,
                30.60000038,
            ]
        elif degree < 70:
            bias_table[bias_level] = [
                8.199999809,
                12.47999992,
                13.91999969,
                15.39999962,
                16.89999962,
                18.29999924,
                19.87999954,
                21.5,
                23.79999924,
                35.70000076
            ]
        elif degree <= 90.1:
            bias_table[bias_level] = [
                -15.93999977,
                -1.45999999,
                9.119999695,
                17.27999954,
                22.5,
                24.10000038,
                25.10000038,
                27.52000008,
                30.68000069,
                69.69999695
            ]
        else:
            raise ValueError(f"{degree},{bias_level})Over 90 degree.+get_reflect_closure")
        #else:
        bias_level = str(bias_table["key"][4]) #"1000"
        if degree < 30:
            bias_table[bias_level] = [
                -2.25999999,
                0.699999988,
                2.299999952,
                3.200000048,
                4.400000095,
                5.599999905,
                6.780000114,
                8,
                9.5,
                23.79999924
            ]
        elif degree < 45:
            bias_table[bias_level] = [
                0,
                5.599999905,
                7.800000191,
                9.100000381,
                10,
                10.80000019,
                12,
                13.69999981,
                16.5,
                35.5
            ]
        elif degree < 80:
            bias_table[bias_level] = [
                -1.170000041,
                6.900000095,
                11.10000038,
                13.41999969,
                15.19999981,
                17.20000076,
                19,
                20.74000015,
                23.57000027,
                53.20000076
            ]
        elif degree <= 90.1:
            bias_table[bias_level] = [
                -15.19999981,
                -6.900000095,
                -4.400000095,
                0.100000001,
                3.799999952,
                9.100000381,
                17.89999962,
                21.5,
                25.89999962,
                40.5,
            ]
        else:
            raise ValueError(f"{degree},{bias_level})Over 90 degree.+get_reflect_closure")
        
        def simple_reflect_ball2wall(data:dict):
            # reflect vec 구하기
            #   방향벡터를 반대로 한 벡터에서 노멀벡터를 기준으로 반전한 벡터가 기본적인 반사 벡터이다.
            x , y = normal_vec.tolist()
            reflect_vec = reverse_direct_vec.copy()
            if x==0:
                reflect_vec[0] = -reverse_direct_vec[0]
            else:
                reflect_vec[1] = -reverse_direct_vec[1]
            
            # power, sidespin 힘
            power = data["power"]
            sidespin = data["sidespin"]
            # upspin = data["upspin"]
            # upspin_lv = int((upspin - upspinmin) / (upsinrange) * 10)
            
            #power에 따른 좌우스핀에 따른 반사각 편이의 정도를 담은 테이블 선택
            for speed_guide in bias_table["key"]:
                if float(power) < float(speed_guide):
                    test_print("speed_guide",power,speed_guide)
                    pick_key = speed_guide
            try:
                table = bias_table[str(pick_key)]
            except:
                print("error degree : ",degree, str(pick_key))
            # test_print("===============table=============\n", table)
            
            
            # sidespin level에 따른 반사각의 편이 정도 선택
            if sidespin == 0.0:
                theta = 0
            #   부딪치는 방향에 따라 정회전인지 역회전인지 확인
            else : 
                #정회전인 경우
                if sidespin * right_side > 0: # right_side에서는 정방향이 어느 방향인지 담고 있다. 같은 부호를 가르킨다면 정수가 나올 것이다.
                    sidespin_lv = int((sidespin - sidespinmin) / (sidespinrange) * 10)
                    if sidespin_lv >= 10:
                        sidespin_lv = 9 
                else:#역회전
                    sidespin_lv = int(0)
                # set new degree
                theta = table[sidespin_lv]
            
            # right_side는 입사하는 방향벡터와 노멀벡터와의 사이와 같다.
            # 노멀벡터를 기준으로 방향벡터는 대칭하여 같은 값을 가진다.
            # 하지만 회전연산은 서로 다르다.
            # right_side를 통해서 방향벡터의 입사방향을 판단할 수 있도록 한다.
            # 입사 방향이 반대라면, 회전연산이 역으로 걸릴 수 있도록 해준다.
            reflect_vec = rotate_vector(reflect_vec, theta, reverse_rotate=(right_side > 0))
            
            return (reflect_vec, data)
        return simple_reflect_ball2wall
    
    def update(self, event:dict=None) -> None:
        pass
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