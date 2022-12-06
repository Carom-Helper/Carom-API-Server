import cv2
import numpy as np
from pipe_cls import One2OnePipe
from detect_utills import PipeResource, is_test

def is_test_image_rotate()->bool:
    return True and is_test()

# set path
import os
from pathlib import Path
import sys
CAROM_BASE_DIR=Path(__file__).resolve().parent.parent.parent

def check_aline_corners(width, top_left, top_right, bottom_left, bottom_right)-> bool:
    # top_left, top_right, bottom_left, bottom_right 들의 배치가 잘 맞는지 확인해 주는 함수
    origin_list = [top_left, top_right, bottom_left, bottom_right]
    sorted_list = [top_left, top_right, bottom_left, bottom_right]
    sorted_list.sort(key=lambda x:x[0] + x[1]*width)
    
    print("origin_list : ",origin_list)
    print("sorted_list : ",sorted_list)
    for origin, sorted in zip(origin_list,sorted_list):
        if origin != sorted:
            return False
    return True
    
def aline_corner_in_dict(metadata:dict)->list:
    # 들어온 값을 정렬하여서 [TL, TR, BL, BR] 순서로 반환한다.
    pts = list()
    width = metadata['WIDTH']
    for key, value in metadata.items():
        if key == 'TL' or key == 'BL' or key == 'TR' or key == 'BR':
            pts.append(value)
    pts.sort(key=lambda x:x[0] + x[1]*width)
    pts = [pts[0],pts[1],pts[3],pts[2]]
    return pts

def rotate_origin_90(xy, shift_len, right_direct=False):
    import math 
    
    x_shift = 1 if right_direct else 0
    y_shift =  0 if right_direct else 1
    
    degree = 90 
    theta = math.radians(degree)
    
    cos = 0 #math.cos(theta) * (1 if right_direct else -1)
    sin = 1 if right_direct else -1 #math.sin(theta)
    
    """Only rotate a point around the origin (0, 0)."""
    x, y = xy
    xx = x * cos - y * sin + shift_len * x_shift
    yy = x * sin + y * cos - shift_len * y_shift

    return int(xx), int(yy)

def fit_resizer(img:np.ndarray)->np.ndarray:
    width = 1440
    height = 1080
    h, w, _ = img.shape
    
    proportion = 1.0
    w_proportion = h_proportion = 1.0
    if w > width:
        w_proportion = width / w
    if h > height:
        h_proportion = height/ h
    
    proportion = w_proportion if w_proportion < h_proportion else h_proportion
    result = cv2.resize(img, (0,0), fx=proportion, fy=proportion)
    return result
    

def line_show(img:np.ndarray, conners:list, name="defalut"):
    result = img.copy()
    
    points = [conners[0],conners[1],conners[3],conners[2]]
    color_list = [(127, 0, 225),(127, 127, 0),(255, 127, 0),(255, 255, 225)]
    
    
    for i in range(4):
        result = cv2.line(result, 
                (int(points[i][0]), int(points[i][1])), 
                (int(points[(i+1)%4][0]), int(points[(i+1)%4][1])), 
                (0, 255, 0), 2)
    
    for i in range(4):
        result = cv2.circle(result, (points[i][0], points[i][1]), 9, color_list[i], 5)
    
    result = fit_resizer(result)
    cv2.imshow(name, result)

class ImageRotationPipe(One2OnePipe):
    def __init__(self) -> None:
        super().__init__()
    
    def exe(self, input: PipeResource) -> PipeResource:
        output = input
        
        
        
        
        
        return output
    
    
    def get_regist_type(self, idx=0) -> str:
        return "image_rotater"


def test_check_aline():
    max = 10
    
    sample_path = np.fromfile(CAROM_BASE_DIR / "media" / "test2" / "sample.jpg", np.uint8)
    sample = cv2.imdecode(sample_path, cv2.IMREAD_COLOR)
    
    ca3655_path = np.fromfile(CAROM_BASE_DIR / "media" / "carom" / "CAP3825091495947943655.jpg ", np.uint8)
    ca3655 = cv2.imdecode(ca3655_path, cv2.IMREAD_COLOR)
    ca3655 = cv2.resize(ca3655, (720,1280))
    
    #sample image test
    sample_points = [[549, 109], [942, 111], [180, 565], [1270, 580]] # sample
    sample_wh=[1440, 681]
    
    # CAP3825091495947943655 image test
    ca3655_points = [[662, 452], [662, 752], [57, 147], [57, 1057]] # CAP3825091495947943655.jpg
    
    ca3655_wh=[720,1280]
    
    now_sample_points = sample_points.copy()
    now_ca3655_points = ca3655_points.copy()
    
    print(now_sample_points)
    print(f"seample.jpg : {check_aline_corners(sample_wh[0], *now_sample_points)}")
    
    print(now_ca3655_points)
    print(f"CAP3825091495947943655.jpg : {check_aline_corners(ca3655_wh[0], *now_ca3655_points)}")
    
    line_show(sample, now_sample_points, "sample.jpg")
    line_show(ca3655, now_ca3655_points, "CAP3825091495947943655.jpg")
    cv2.waitKey()
    
    for cnt in range(1, max+1):
        
        direct = True
        
        if direct:
            idx = (cnt)%2
        else:
            idx = (cnt+1)%2
            
        sample_shift_len = sample_wh[idx]
        ca3655_shift_len = ca3655_wh[idx]
        # rotation
        target_list = [now_sample_points, now_ca3655_points]
        target_shift_len_list = [sample_shift_len, ca3655_shift_len]
        
        
        # conner point rotation
        for target_num in range(2):
            target = target_list[target_num]
            shfit_len = target_shift_len_list[target_num]
            for idx in range(4):
                x, y = rotate_origin_90(target[idx], shfit_len, direct)
                target[idx] = [x,y]
                
            if direct:
                target = [target[2],target[0],target[3],target[1]]
            else :
                target = [target[1],target[3],target[0],target[2]]

        # image rotation
        img_direct = cv2.ROTATE_90_CLOCKWISE if direct else cv2.ROTATE_90_COUNTERCLOCKWISE
        
        sample = cv2.rotate(sample, img_direct)
        ca3655 = cv2.rotate(ca3655, img_direct)
        
        # view result
        print(f"=================== Rotation {cnt}s ====================")
        print(now_sample_points)
        print(f"seample.jpg : {check_aline_corners(sample_shift_len, *now_sample_points)}")
        print(now_ca3655_points)
        print(f"CAP3825091495947943655.jpg : {check_aline_corners(ca3655_shift_len, *now_ca3655_points)}")
        print("======================================================")
        line_show(sample, now_sample_points, "sample.jpg")
        line_show(ca3655, now_ca3655_points, "CAP3825091495947943655.jpg")
        cv2.waitKey()
        
        
def test_line_show():
    sample_path = np.fromfile(CAROM_BASE_DIR / "media" / "test2" / "sample.jpg", np.uint8)
    sample = cv2.imdecode(sample_path, cv2.IMREAD_COLOR)
    
    ca3655_path = np.fromfile(CAROM_BASE_DIR / "media" / "carom" / "CAP3825091495947943655.jpg ", np.uint8)
    ca3655 = cv2.imdecode(ca3655_path, cv2.IMREAD_COLOR)
    ca3655 = cv2.resize(ca3655, (720,1280))
    
    sample_points = [[549, 109], [942, 111], [180, 565], [1270, 580]] # sample
    ca3655_points = [[57, 147], [662, 452], [57, 1057], [662, 752]] # sample
    
    line_show(sample, sample_points, "sample.jpg")
    line_show(ca3655, ca3655_points, "CAP3825091495947943655.jpg")
    cv2.waitKey()

def runner(args):
    # test_line_show()
    test_check_aline()
    
if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--src', default=CAROM_BASE_DIR / "media" / "carom" / "CAP3825091495947943655.jpg")
    # parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--no_display', default=False, action="store_true")
    # args = parser.parse_args()
    args = 0
    runner(args)