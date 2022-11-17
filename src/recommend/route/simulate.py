import argparse
import numpy as np
import cv2

from route_utills import is_test, print_args
from action_cls import *
from CaromBall import CaromBall, set_vec
from WallObject import WallObject

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

    wall_list = list()
    wall_list.append(WallObject(
            pos1={"x":0.0, "y":0.0}, 
            pos2={"x":400.0, "y":0.0},
            name="top"
        ))
    wall_list.append(WallObject(
            pos1={"x":400.0, "y":0.0}, 
            pos2={"x":400.0, "y":800.0},
            name="right"
        ))
    wall_list.append(WallObject(
            pos1={"x":400.0, "y":800.0}, 
            pos2={"x":0.0, "y":800.0},
            name="bottom"
        ))
    wall_list.append(WallObject(
            pos1={"x":0.0, "y":800.0}, 
            pos2={"x":0.0, "y":0.0},
            name="left"
        ))
    w1 = wall_list[0]
    w2 = wall_list[1]
    w3 = wall_list[2]
    w4 = wall_list[3]

    cue.start_param(power = power, clock = clock, tip = tip)
    cue.print_param()

    cue.set_xy(*cue_coord)
    tar1.set_xy(*tar1_coord)
    tar2.set_xy(*tar2_coord)

    #cue.register_observer(tar1)
    #cue.register_observer(tar2)

    cue.register_observer(w1)
    cue.register_observer(w2)
    cue.register_observer(w3)
    cue.register_observer(w4)
    
    w1.register_observer(cue)
    w2.register_observer(cue)
    w3.register_observer(cue)
    w4.register_observer(cue)

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
    
    # for c in range(12):
    #     for t in range(1,4):
    #         for p in range(10, 60, 10):
    #             for th in range(-7, 8):
    #                 # c = 5
    #                 # p = 50
    #                 # th = 0
    #                 print(c, t, p, th)
    #                 #test(args.cue, args.tar1, power=p, clock=c, tip=t, thick=th)
    #                 test(power=p, clock=c, tip=t, thick=th)
    
    test(power = 50, clock = 0, tip = 3, thick = 0)
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