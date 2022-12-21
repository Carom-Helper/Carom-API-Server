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
    save=False,
    debuging=False
    ):
    
    # create object(ball, wall)
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

    # ball init
    cue.start_param(power = power, clock = clock, tip = tip)
    cue.set_xy(*cue_coord)
    tar1.set_xy(*tar1_coord)
    tar2.set_xy(*tar2_coord)
    
    
    # 충돌객체 추가
    #   공 <- 벽 객체 추가
    for wall in wall_list:
        cue.add_crashable_object(wall)
        tar1.add_crashable_object(wall)
        tar2.add_crashable_object(wall)
    
    #   공 <- 다른 공 객체 추가
    cue.add_crashable_object(tar1)
    cue.add_crashable_object(tar2)
    tar1.add_crashable_object(tar2)
    
    set_vec(cue, tar1, thick)
    cue.set_mover(cue.move_by_time)
    tar1.set_mover(tar1.move_stay)
    tar2.set_mover(tar2.move_stay)

    elapsed = 1
    
    success = False
    wall_count = 0
    is_tar1_hit = False
    is_tar2_hit = False
    
    ball_list = [cue, tar1, tar2]
    while True:
        # ball move
        cue_dist, _ = cue.move(elapsed)
        tar1_dist, _ = tar1.move(elapsed)
        tar2_dist, _ = tar2.move(elapsed)
        
        def lazy_set_crash_action():
            cue.check_crash_event_and_notify_event_to_observers()
            tar1.check_crash_event_and_notify_event_to_observers()
            tar2.check_crash_event_and_notify_event_to_observers()
        lazy_set_crash_action()
        
        cue.apply_next_action()
        tar1.apply_next_action()
        tar2.apply_next_action()
        

        elapsed = min(cue_elapsed, tar1_elapsed, tar2_elapsed)

        cue_hit = cue.crash_objects
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
            print(success, cue.crash_points)
    name = f"({cue_coord[0]}cue{cue_coord[1]})({tar1_coord[0]}tar{tar1_coord[1]})({tar2_coord[0]}tar{tar2_coord[1]})(P{power}C{clock}T{tip})(thick{thick}).jpg"
    show(cue, tar1, tar2, name, display=display, save = debuging or (success and save))

    return success, cue.crash_points, tar1.crash_points, tar2.crash_points

def show(cue, tar1, tar2, name, display=True, save=False):
    img = np.zeros((800,400,3), np.uint8)
    clist = cue.xy
    img = cv2.line(img, (int(clist[0]['x']), int(clist[0]['y'])), (int(clist[0]['x']), int(clist[0]['y'])), (0, 255, 255), 6)
    t1list = tar1.xy
    img = cv2.line(img, (int(t1list[0]['x']), int(t1list[0]['y'])), (int(t1list[0]['x']), int(t1list[0]['y'])), (0, 0, 255), 6)
    t2list = tar2.xy
    img = cv2.line(img, (int(t2list[0]['x']), int(t2list[0]['y'])), (int(t2list[0]['x']), int(t2list[0]['y'])), (0, 255, 0), 6)

    for c in clist:
        img = cv2.line(img, (int(c['x']), int(c['y'])), (int(c['x']), int(c['y'])), (0, 255, 255), 2)
    for t in t1list:
        img = cv2.line(img, (int(t['x']), int(t['y'])), (int(t['x']), int(t['y'])), (0, 0, 255), 2)
    for t in t2list:
        img = cv2.line(img, (int(t['x']), int(t['y'])), (int(t['x']), int(t['y'])), (0, 255, 0), 2)
    
    if save:
        print("save : ", name)
        cv2.imwrite(name ,img)
    try:
        if display:
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
    debuging=False,
    success_list = [],
    clock = [0],
    tip_sequence = [3],
    power_sequence = [40, 50],
    think_sequence = [-4, 4, -3, 3, -2, 2, -5, 5, -6, 6, -7, 7, -1, 1]
    ):
    for th in think_sequence: # think 단위로 하나씩 결과를 뽑는다
        for t in tip_sequence:
            for p in power_sequence:
                for c in clock:
                    if len(success_list) >= DETECT_ROUTE_NUM:
                        return success_list
                    success, cue, tar1, tar2 = run_carom_simulate(cue_coord=cue_coord,
                                                    tar1_coord=tar1_coord,
                                                    tar2_coord=tar2_coord,
                                                    power=p,
                                                    clock=c,
                                                    tip=t,
                                                    thick=th,
                                                    display=display,
                                                    save=save,
                                                    debuging=debuging)
                    if len(success_list) >= DETECT_ROUTE_NUM:
                        return
                    if success:
                        result = {"power": p,
                                    "clock": c,
                                    "tip": t,
                                    "thick": th,
                                    "cue": cue, 
                                    "tar1": tar1,
                                    "tar2": tar2}
                        success_list.append(result)
                        print("\nsimulate_thread : ",result)
                        return

def simulation(
    cue_coord=(300,400), 
    tar1_coord=(100,750),
    tar2_coord=(300,300),
    display=True,
    save=False,
    debuging=False,
    is_thread = True,
    clock_sequence = [1, 11, 2, 10, 0, 3, 9, 4, 8],
    tip_sequence = [3,2],
    power_sequence = [30, 40, 50],
    think_sequence = [-4, 4, -3, 3, -2, 2, -5, 5, -6, 6, -7, 7, -1, 1]
    ):

    success_list = []
    success_clock = []
    thread_list = []
    # clock_sequence = [3]
    # for c in clock_sequence:
    
    tar_ball = [tar1_coord, tar2_coord]
    for th in think_sequence:
        for tar1_idx in range(2):
            if is_thread:
                thread = threading.Thread(target=simulate_thread, args=(
                    cue_coord,
                    tar_ball[tar1_idx],
                    tar_ball[1-tar1_idx],
                    display,
                    save,
                    debuging,
                    success_list,
                    clock_sequence,
                    tip_sequence,
                    power_sequence,
                    [th]))
                thread.start()
                thread_list.append(thread)
            else :
                simulate_thread(
                    cue_coord,
                    tar_ball[tar1_idx],
                    tar_ball[1-tar1_idx],
                    display,
                    save,
                    debuging,
                    success_list,
                    clock_sequence,
                    tip_sequence,
                    power_sequence,
                    [th])
    if is_thread:
        for thr in thread_list:
            thr.join()
        
    print("\nsimulation : ",success_list)
    return success_list


def tolist_by_int(values)-> list:
    from ast import literal_eval
    if isinstance(values, str):
        values=list(map(int, values.split(' ')))
        
    if isinstance(values, tuple):
        values = list(map(int, values))
    elif isinstance(values, list):
        values = list(map(int, values))
    else:
        raise TypeError
    return values

 
def runner(args):
    print_args(vars(args))
    cue_xy = tuple(tolist_by_int(args.cue))
    tar1_xy = tuple(tolist_by_int(args.tar1))
    tar2_xy = tuple(tolist_by_int(args.tar2))
    
    clock_sequence = tolist_by_int(args.clock)
    tip_sequence = tolist_by_int(args.tip)
    power_sequence = tolist_by_int(args.power)
    think_sequence = tolist_by_int(args.think)
    
    print("Start Simulate : ", f"[ cue:{cue_xy} | tar1:{tar1_xy} | tar2:{tar2_xy} ]")
    print(f"Clock = {clock_sequence}")
    print(f"Tip = {tip_sequence}")
    print(f"Power = {power_sequence}")
    print(f"Think = {think_sequence}")
    simulation(cue_coord=cue_xy,
               tar1_coord=tar1_xy,
               tar2_coord=tar2_xy,
               display=False,
               save=not args.no_save,
               debuging=args.debug,
               is_thread = not args.no_thread,
               clock_sequence = clock_sequence,
               tip_sequence = tip_sequence,
               power_sequence = power_sequence,
               think_sequence = think_sequence)

    #run(args.src, args.device)
    # detect(args.src, args.device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cue', nargs="+", default="300 400", help="--cue x y")
    parser.add_argument('--tar1', nargs="+", default="100 750", help="--tar1 x y")
    parser.add_argument('--tar2', nargs="+", default="300 300", help="--tar2 x y")
    parser.add_argument('--clock', nargs="+", default="0", help="--clock 1 11 2 10") 
    parser.add_argument('--tip', nargs="+", default="3 2", help="--tip 3 1")
    parser.add_argument('--power', nargs="+", default="40", help="--power 20 30")
    parser.add_argument('--think', nargs="+", default="-6 -7", help="--thick '-4 4'")
    # parser.add_argument('--clock', nargs="+", default="1 11 2 10 0 3 9 4 8", help="--clock 1 11 2 10") 
    # parser.add_argument('--tip', nargs="+", default="3 2", help="--tip 3 1")
    # parser.add_argument('--power', nargs="+", default="30 40 50", help="--power 20 30")
    # parser.add_argument('--think', nargs="+", default="-4 4 -5 5 -6 6 -3 3 -2 2 -1 1 -7 7", help="--thick '-4 4'")
    parser.add_argument('--debug', default=False, action="store_true")
    parser.add_argument('--no_thread', default=False, action="store_true")
    parser.add_argument('--no_save', default=False, action="store_true")
    
    # parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--display', action="store_true")
    args = parser.parse_args()
    runner(args) 