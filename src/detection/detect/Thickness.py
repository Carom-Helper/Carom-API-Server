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

    projected = np.zeros((800,400,3), np.uint8)
    projected = cv2.circle(projected, (int(cue['x']), int(cue['y'])), 8, (255, 255, 255), 1)
    projected = cv2.circle(projected, (int(tar['x']), int(tar['y'])), 8, (0, 0, 255), 1)
    projected = cv2.circle(projected, (int(new_t['x']), int(new_t['y'])), 8, (0, 255, 0), 1)

    print(f'{int(thickness*8)}/8')
    cv2.imshow('prediction', projected)
    cv2.waitKey()

cue = {'x':300, 'y':400}
tar = {'x':350, 'y':350}
for i in range(-8, 9):
    thickness_prediction(cue, tar, i/8)