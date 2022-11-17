import argparse
import cv2
import numpy as np

from route_utills import is_test, print_args
from action_cls import CaromBall, set_vec

def is_test_caromball()->bool:
    return True and is_test()

def test_print(s, s1="", s2="", s3="", s4="", s5="", end="\n"):
    if is_test_caromball():
        print("action cls exe : ", s, s1, s2, s3, s4, s5, end=end)

def test():
    cue = CaromBall()
    tar1 = CaromBall()
    tar2 = CaromBall()

    cue.set_xy(300,400)
    tar1.set_xy(100,750)
    tar2.set_xy(300,350)

    cue.start_param(clock = 12, tip = 1)
    cue.print_param()

    cue.register_observer(tar1)
    cue.register_observer(tar2)

    set_vec(cue, tar1, 7)
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