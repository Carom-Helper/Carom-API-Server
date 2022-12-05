import argparse
import numpy as np
import cv2
import threading

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
    display=True,
    save=False
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

    tar1.register_observer(w1)
    tar1.register_observer(w2)
    tar1.register_observer(w3)
    tar1.register_observer(w4)
    tar1.register_observer(cue)
    tar1.register_observer(tar2)
    
    tar2.register_observer(w1)
    tar2.register_observer(w2)
    tar2.register_observer(w3)
    tar2.register_observer(w4)
    tar2.register_observer(cue)
    tar2.register_observer(tar1)

    w1.register_observer(cue)
    w2.register_observer(cue)
    w3.register_observer(cue)
    w4.register_observer(cue)
    
    set_vec(cue, tar1, thick)
    cue.set_mover(cue.move_by_time)
    tar1.set_mover(tar1.move_stay)
    tar2.set_mover(tar2.move_stay)

    elapsed = 1
    
    success = False
    wall_count = 0
    is_tar1_hit = False
    is_tar2_hit = False
    while True:
    # for _ in range(10000):
        cue_dist, _ = cue.move(elapsed)
        tar1_dist, _ = tar1.move(elapsed)
        tar2_dist, _ = tar2.move(elapsed)

        for observer in cue.observer_list:
            cue.notify_filltered_observer(observer)
            
        for observer in tar1.observer_list:
            tar1.notify_filltered_observer(observer)
            
        for observer in tar2.observer_list:
            tar2.notify_filltered_observer(observer)
        
        cue_elapsed = cue.update()
        tar1_elapsed = tar1.update()
        tar2_elapsed = tar2.update()

        elapsed = min(cue_elapsed, tar1_elapsed, tar2_elapsed)

        cue_hit = cue.crash_list
        if cue_hit is not None:
            for hit in cue_hit:
                if hit == 'tar1':
                    is_tar1_hit = True
                elif hit == 'tar2':
                    is_tar2_hit = True
                else:
                    wall_count += 1
            
        if is_tar1_hit and is_tar2_hit:
            if wall_count >= 3:
                success = True
            break

        if cue_dist < 0.0005 and tar1_dist < 0.0005 and tar2_dist < 0.0005:
            break

        if tar2_dist > 0:
            break
    if display:
        if success:
            print(success, cue.colpoint)
        name = f"({cue_coord[0]}-{cue_coord[1]})({tar1_coord[0]}-{tar1_coord[1]})({power}-{clock}-{tip})({thick}).jpg"
        show(cue, tar1, tar2, name, success and save)

    return success, cue.colpoint, tar1.colpoint, tar2.colpoint

def show(cue, tar1, tar2, name, save=False):
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
    
    if save:
        print("save : ", name)
        cv2.imwrite(name ,img)
    try:
        cv2.imshow('simulate', img)
        cv2.waitKey(1000)
    except InterruptedError:
        raise InterruptedError()
    except:pass

def show_ani(cue, tar1, tar2):
    img = np.zeros((800,400,3), np.uint8)
    clist = cue.xy
    img = cv2.line(img, (int(clist[0]['x']), int(clist[0]['y'])), (int(clist[0]['x']), int(clist[0]['y'])), (255, 255, 255), 3)
    t1list = tar1.xy
    img = cv2.line(img, (int(t1list[0]['x']), int(t1list[0]['y'])), (int(t1list[0]['x']), int(t1list[0]['y'])), (0, 0, 255), 3)
    t2list = tar2.xy
    img = cv2.line(img, (int(t2list[0]['x']), int(t2list[0]['y'])), (int(t2list[0]['x']), int(t2list[0]['y'])), (0, 255, 0), 3)
    i=1
    while i < min(len(clist), len(t1list), len(t2list)):
        for c in clist[:i]:
            img = cv2.line(img, (int(c['x']), int(c['y'])), (int(c['x']), int(c['y'])), (255, 255, 255), 1)
        for t in t1list[:i]:
            img = cv2.line(img, (int(t['x']), int(t['y'])), (int(t['x']), int(t['y'])), (0, 0, 255), 1)
        for t in t2list[:i]:
            img = cv2.line(img, (int(t['x']), int(t['y'])), (int(t['x']), int(t['y'])), (0, 255, 0), 1)
        try:
            cv2.imshow('simulate', img)
            cv2.waitKey(1)
        except:pass
        i+=1
    
def simulate_thread(
    cue_coord=(300,400), 
    tar1_coord=(100,750),
    tar2_coord=(300,300),
    display=True,
    save=False,
    clock = 0,
    success_list = []
    ):

    for t in [3]:
        for p in [40, 50]:
            for th in [-4, 4, -3, 3, -2, 2, -5, 5, -6, 6, -7, 7, -1, 1]:
                    for _ in range(2):
                        if len(success_list) >= DETECT_ROUTE_NUM:
                            return success_list
                        success, cue, tar1, tar2 = run_carom_simulate(cue_coord=cue_coord,
                                                        tar1_coord=tar1_coord,
                                                        tar2_coord=tar2_coord,
                                                        power=p,
                                                        clock=clock,
                                                        tip=t,
                                                        thick=th,
                                                        display=display,
                                                        save=save)
                        if len(success_list) >= DETECT_ROUTE_NUM:
                            return
                        if success:
                            result = {"power": p,
                                        "clock": clock,
                                        "tip": t,
                                        "thick": th,
                                        "cue": cue, 
                                        "tar1": tar1,
                                        "tar2": tar2}
                            success_list.append(result)
                            print(success_list)
                            return
                        else:
                            temp = tar1_coord
                            tar1_coord = tar2_coord
                            tar2_coord = temp

def simulation(
    cue_coord=(300,400), 
    tar1_coord=(100,750),
    tar2_coord=(300,300),
    display=True,
    save=False
    ):

    success_list = []
    success_clock = []
    thread_list = []
    clock_sequence = [1, 2, 11, 10, 0, 9, 3, 4, 8]
    for c in clock_sequence:
        thread = threading.Thread(target=simulate_thread, args=(
            cue_coord,
            tar1_coord,
            tar2_coord,
            display,
            save,
            c,
            success_list))
        thread.start()
        thread_list.append(thread)

    for thr in thread_list:
        print('waiting for ', thr)
        thr.join()
        
    print(success_list)
    return success_list
    """
    #test area ###################################################################################
    p, c, t, th = 50, 2, 3, 0
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
                    "thick": th,
                    "cue": cue, 
                    "tar1": tar1,
                    "tar2": tar2}
        success_list.append(result)
        success_clock.append(c)
        if len(success_list) >= DETECT_ROUTE_NUM:
            return success_list
    #test area ###################################################################################
    """
    for c in clock_sequence:
        for t in [3]:
            for p in [40, 50]:
                for th in [-4, 4, -3, 3, -2, 2, -5, 5, -6, 6, -7, 7, -1, 1]:
                    for _ in range(2):
                        success, cue, tar1, tar2 = run_carom_simulate(cue_coord=cue_coord,
                                                        tar1_coord=tar1_coord,
                                                        tar2_coord=tar2_coord,
                                                        power=p,
                                                        clock=c,
                                                        tip=t,
                                                        thick=th,
                                                        display=display,
                                                        save=save)
                        if success:
                            result = {"power": p,
                                        "clock": c,
                                        "tip": t,
                                        "thick": th,
                                        "cue": cue, 
                                        "tar1": tar1,
                                        "tar2": tar2}
                            success_list.append(result)
                            success_clock.append(c)
                            if len(success_list) >= DETECT_ROUTE_NUM:
                                return success_list
                        
                        temp = tar1_coord
                        tar1_coord = tar2_coord
                        tar2_coord = temp

                    if c in success_clock:
                        break
                if c in success_clock:
                    break
            if c in success_clock:
                break

    for c in [x for x in clock_sequence if x not in success_clock]:
        for t in [2, 3]:
            for th in [-5, 5, -6, 6, -7, 7, -4, 4, -3, 3, -2, 2]:
                if t == 3 and abs(th) < 5:
                    pass
                for _ in range(2):
                    success, cue, tar1, tar2 = run_carom_simulate(cue_coord=cue_coord,
                                                        tar1_coord=tar1_coord,
                                                        tar2_coord=tar2_coord,
                                                        power=40,
                                                        clock=c,
                                                        tip=t,
                                                        thick=th,
                                                        display=display,
                                                        save=save)
                    if success:
                        result = {"power": p,
                                    "clock": c,
                                    "tip": t,
                                    "thick": th,
                                    "cue": cue, 
                                    "tar1": tar1,
                                    "tar2": tar2}
                        success_list.append(result)
                        success_clock.append(c)
                        if len(success_list) >= DETECT_ROUTE_NUM:
                            print(success_list)
                            return success_list
                    
                    temp = tar1_coord
                    tar1_coord = tar2_coord
                    tar2_coord = temp

                    if c in success_clock:
                        break
                if c in success_clock:
                    break
    

    print(success_list)
    return success_list

def get_ball_coord(ball):
    from ast import literal_eval
    if isinstance(ball, str):
        x,y=map(int, ball.split(','))
        ball = (x,y)
    
    
    if isinstance(ball, tuple):
        pass
    elif isinstance(ball, list):
        x,y = ball
        ball = (x,y)
    else:
        raise TypeError
    return ball
def runner(args):
    print_args(vars(args))
    cue_xy = get_ball_coord(args.cue)
    tar1_xy = get_ball_coord(args.tar1)
    tar2_xy = get_ball_coord(args.tar2)
    simulation(cue_coord=cue_xy, tar1_coord=tar1_xy, tar2_coord=tar2_xy, display=False, save=args.save)

    #run(args.src, args.device)
    # detect(args.src, args.device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cue', default=(300,400))
    parser.add_argument('--tar1', default=(100,750))
    parser.add_argument('--tar2', default=(300,300))
    parser.add_argument('--save', default=False, action="store_true")
    
    # parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--display', action="store_true")
    args = parser.parse_args()
    runner(args) 