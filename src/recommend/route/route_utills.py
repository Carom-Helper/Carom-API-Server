def is_test()->bool:
    return False

def test_print(s, s1="", s2="", s3="", s4="", s5="", end="\n"):
    if is_test():
        print("utills cls exe : ", s, s1, s2, s3, s4, s5, end=end)

import cv2
import numpy as np

import sys
from pathlib import Path
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parent  # YOLOv5 root directory
CAROM_BASE_DIR = ROOT.parent.parent

from typing import Optional
import inspect


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    # Print function arguments (optional args dict)
    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix('')
    except ValueError:
        file = Path(file).stem
    s = (f'{file}: ' if show_file else '') + (f'{func}: ' if show_func else '')
    print(colorstr(s) + ', '.join(f'{k}={v}' for k, v in args.items()))
    
    
def thickness_prediction(cue, tar, thickness = 0):
    radius = 8.6

    cue_tar = {'x':(cue['x'] - tar['x']), 'y':(cue['y'] - tar['y'])}
    x = thickness * radius
    y = (radius**2 - x**2)**0.5

    x *= 1.5
    y *= 1.5

    new_t = {'x':x, 'y':y}

    cue_tar_l = (cue_tar['x']**2+cue_tar['y']**2)**0.5
    cos = cue_tar['y'] / cue_tar_l
    sin = -cue_tar['x'] / cue_tar_l


    new_t['x'] = x * cos - y * sin + tar['x']
    new_t['y'] = x * sin + y * cos + tar['y']

    return new_t

# input x,y는 2차원 벡터
# return 되는 각도는 radian
def angle(x,y):
    try:
        v=np.inner(x,y) / (np.linalg.norm(x) * np.linalg.norm(y))
        theta = np.arccos(v)
        return theta
    except ZeroDivisionError:
        print("angle : divide zero")
        return 0
    except:
        return 0
def radian2degree(theta):
    return np.rad2deg(theta)

def test_degree():
    from random import random
    normal_vec_list = list()
    normal_vec_list.append(np.array([0.0,-1.0])) # top
    normal_vec_list.append(np.array([-1.0,0.0])) # right
    normal_vec_list.append(np.array([0.0,1.0])) # bottom
    normal_vec_list.append(np.array([1.0,0.0])) # left
    
    x = random()
    direct_vec = [x, 1.0-x]
    direct_vec = np.array(direct_vec)
    print("direct_vect", direct_vec)
    for normal_vec in normal_vec_list:
        theta = angle(normal_vec, direct_vec)
        theta = radian2degree(theta)
        print(f"direct({direct_vec})/normal({normal_vec}) : {theta}")

def test():
    cue = {'x':300, 'y':400}
    tar = {'x':350, 'y':350}
    for i in range(-8, 9):
        new_t = thickness_prediction(cue, tar, i/8)
        projected = np.zeros((800,400,3), np.uint8)
        projected = cv2.circle(projected, (int(cue['x']), int(cue['y'])), 8, (255, 255, 255), 1)
        projected = cv2.circle(projected, (int(tar['x']), int(tar['y'])), 8, (0, 0, 255), 1)
        projected = cv2.circle(projected, (int(new_t['x']), int(new_t['y'])), 8, (0, 255, 0), 1)
        print(f'{i}/8')
        cv2.imshow('prediction', projected)
        cv2.waitKey()
    
if __name__ == '__main__':
    test_degree()
    #test()