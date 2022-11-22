import argparse
import numpy as np
import cv2

from route_utills import is_test, print_args
from action_cls import *
from CaromBall import CaromBall, set_vec
from WallObject import WallObject

DETECT_ROUTE_NUM=3

def run_carom_simulate(
    cue_coord=(300,400), 
    tar1_coord=(100,750),
    tar2_coord=(300,300),
    power = 50,
    clock = 12,
    tip = 1,
    thick = 0,
    display=True
    ):
    cue = CaromBall("cue")
    tar1 = CaromBall("tar1")
    tar2 = CaromBall("tar2")

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
    #cue.print_param()

    cue.set_xy(*cue_coord)
    tar1.set_xy(*tar1_coord)
    tar2.set_xy(*tar2_coord)

    #cue.register_observer(tar1)
    #cue.register_observer(tar2)

    cue.register_observer(w1)
    cue.register_observer(w2)
    cue.register_observer(w3)
    cue.register_observer(w4)
    cue.register_observer(tar1)
    cue.register_observer(tar2)
    
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
    success = False
    wall_count = 0
    is_tar1_hit = False
    is_tar2_hit = False
    #while True:
    for _ in range(100000):
        cue_hit, cue_dist, cue_elapsed = cue.move(elapsed)
        _, tar1_dist, tar1_elapsed = tar1.move(elapsed)
        _, tar2_dist, tar2_elapsed = tar2.move(elapsed)

        elapsed = cue_elapsed

        if cue_hit is not None:
            for hit in cue_hit:
                if hit == 'tar1':
                    is_tar1_hit = True
                elif hit == 'tar2':
                    is_tar2_hit = True
                else:
                    wall_count += 1
            
        if is_tar1_hit and is_tar2_hit and wall_count >= 3:
            success = True
            break

        if cue_dist < 0.0005 and tar1_dist < 0.0005 and tar2_dist < 0.0005:
            break
    #print(success)
    #print(cue.colpoint)
    #if False:
    if True:
        if display:
            show(cue, tar1, tar2)

    return success, cue.colpoint, tar1.colpoint, tar2.colpoint

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
    try:
        cv2.imshow('simulate', img)
        cv2.waitKey(1000)
    except:pass
    
def simulation(
    cue_coord=(300,400), 
    tar1_coord=(100,750),
    tar2_coord=(300,300),
    display=True
    ):

    success_list = []

    for c in range(12):
        for t in range(1,4):
            for p in range(10, 60, 10):
                for th in range(-7, 8):
                    for _ in range(2):
                        success, cue, tar1, tar2 = run_carom_simulate(cue_coord=cue_coord,
                                                        tar1_coord=tar1_coord,
                                                        tar2_coord=tar2_coord,
                                                        power=p,
                                                        clock=c,
                                                        tip=t,
                                                        thick=th,
                                                        display=display)
                        if success:
                            result = {"power": p,
                                        "clock": c,
                                        "tip": t,
                                        "cue": cue, 
                                        "tar1": tar1,
                                        "tar2": tar2}
                            success_list.append(result)
                            if len(success_list) >= DETECT_ROUTE_NUM:
                                return success_list
                        
                        temp = tar1_coord
                        tar1_coord = tar2_coord
                        tar2_coord = temp
    return success_list

def runner(args):
    print_args(vars(args))
    
    simulation(cue_coord=args.cue, tar1_coord=args.tar1, tar2_coord=args.tar2)

    #run(args.src, args.device)
    # detect(args.src, args.device)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cue', default=(300,400))
    parser.add_argument('--tar1', default=(100,750))
    parser.add_argument('--tar2', default=(300,300))
    # parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--display', action="store_true")
    args = parser.parse_args()
    runner(args) 