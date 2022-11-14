def is_test()->bool:
    return True

def test_print(s, s1="", s2="", s3="", s4="", s5="", end="\n"):
    if is_test():
        print("pipe cls exe : ", s, s1, s2, s3, s4, s5, end=end)

import cv2
import numpy as np

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
    test()