import cv2
import numpy as np

# set path
import sys
from pathlib import Path
import os

CAROM_BASE_DIR=Path(__file__).resolve().parent

from detection.models import *
from u_img.models import *




# 탐지한 모든공 찾기
ball_datas = balls_coord.objects.all().values_list("coord", "carom")
for ball_data in reversed(ball_datas):
    # 탐지한 공 띄우기
    field = np.zeros((800,400,1), np.uint8)
    coord = ball_data[0] # ball_coord.coord
    carom_id = ball_data[1]
    table = carom_data.objects.select_related(
            "img"
            ).values(
                "img_id", 
                "img__img", 
                "guide", 
                "detect_state"
            ).annotate(
                id = F('img_id'),
                img = F('img__img')
            ).get(img_id=carom_id)
    img_path = CAROM_BASE_DIR / "media" / table["img"]
    if not os.path.isfile(img_path) : 
        continue
    guide = table['guide']
    
    
    # 800 * 400 당구공 위치 띄우기
    for key, value in coord.items():
        key = int(key)
        x,y = value
        print("x,y,key:",x,y,key)
        field = cv2.circle(field, (x, y), 9, (255, 255, 255), 1)
    
    # 가이드 라인 띄우기
    img_path = np.fromfile(img_path, np.uint8)
    
    guide_img = cv2.imdecode(img_path, cv2.IMREAD_COLOR)
    guide_img = cv2.resize(guide_img, (720,1280))
    corner_list = list()
    for key, value in guide.items():
        corner_list.append(value)
    corner_list[3], corner_list[2] =corner_list[2], corner_list[3]
    print(corner_list)
    pts = np.zeros((4, 2), dtype=np.float32)
    for i in range(4):
        pts[i] = corner_list[i]
    for i in range(4):
        guide_img = cv2.line(guide_img, (pts[i][0], pts[i][1]), (pts[(i+1)%4][0], pts[(i+1)%4][1]), (0, 255, 0), 2)
    
    
    field = cv2.resize(field, (0,0), fx=0.7, fy=0.7)
    guide_img = cv2.resize(guide_img, (0,0), fx=0.5, fy=0.5)
    cv2.imshow("detect ball", field)
    cv2.imshow("guide image", guide_img)
    # 탐지된 공의 원본 이미지와 가이드라인 그리기
    
    cv2.waitKey(10000)


