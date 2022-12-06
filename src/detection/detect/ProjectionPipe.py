import torch
import numpy as np
import time
import argparse


# set path
import sys
from pathlib import Path
import os

CAROM_BASE_DIR=Path(__file__).resolve().parent.parent.parent
FILE = Path(__file__).resolve()
ROOT = FILE.parent


from pipe_cls import One2OnePipe, ResourceBag
from detect_utills import (PipeResource, LoadImages,
                           aline_corner_in_dict, is_test, cv2, print_args)

def is_test_projection()->bool:
    return True and is_test()

def test_print(s, s1="", s2="", s3="", s4="", s5="", end="\n"):
    if is_test_projection():
        print("projection pipe test : ", s, s1, s2, s3, s4, s5, end=end)

class ProjectionCoordPipe(One2OnePipe):
    def __init__(self, display=True):
        super().__init__()
        self.display = display

    def exe(self, input: PipeResource) -> PipeResource:
        t1 = time.time()

        img = input.im.copy()

        topLeft = input.metadata["TL"]  # x+y가 가장 값이 좌상단 좌표
        bottomRight = input.metadata["BR"]  # x+y가 가장 큰 값이 우하단 좌표
        topRight = input.metadata["TR"]  # x-y가 가장 작은 것이 우상단 좌표
        bottomLeft = input.metadata["BL"]   # x-y가 가장 큰 값이 좌하단 좌표

        # 변환 전 4개 좌표 
        pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])
        
        # pts1 = np.float32(aline_corner_in_dict(input.metadata))

        # 변환 후 4개 좌표
        pts2 = np.float32([[0, 0], [399, 0],
                                [399,799], [0, 799]])

        # 변환 행렬 계산 
        mtrx = cv2.getPerspectiveTransform(pts1, pts2)

        # 원근 변환 적용
        result = cv2.warpPerspective(img, mtrx, (400, 800))
        t2 = time.time()

        projected = np.zeros((800,400,1), np.uint8)
        for det in input.dets:
            temp = np.array([det["x"] + det["w"]/2, det["y"] + det["h"]/2, 1])
            test_print(temp)
            temp_result = mtrx@temp
            projx = int(temp_result[0]/temp_result[2])
            projy = int(temp_result[1]/temp_result[2])
            test_print(temp_result[0]/temp_result[2], temp_result[1]/temp_result[2], "\n")
            result = cv2.line(result, (projx, projy), (projx, projy), (0,0,0), 5)
            projected = cv2.circle(projected, (projx, projy), 9, (255, 255, 255), 1)
            det['x'] = projx
            det['y'] = projy
        
        input.im = result
        
        output = input
        output.set_image('table', projected)
        output.set_image("projected", result)
        
        if self.display :
            print("proj)", end="")
            output.print()
            try:
                cv2.imshow("proj", output.im)
                cv2.imshow("proj2", projected)
            except:pass
        return output

    def get_regist_type(self, idx=0) -> str:
        return "proj_coord"

def coord_test(src, display=True):
    ### Pipe 생성###
    project_pipe = ProjectionCoordPipe(display=display)
    bag_split = ResourceBag()
    
    # 파이프 연결
    project_pipe.connect_pipe(bag_split)
    ### Dataloader ###
    dataset = LoadImages(src)
    ### 실행 ###
    for im0, path, s in dataset:
        width = im0.shape[1]
        hight = im0.shape[0]
        #point 위치 확인
        points = [[549,109],[942,111],[1270,580],[180,565]]
        # points = [[256, 330],[880, 1580],[880, 330],[256, 1580]]
        
        points.sort(key=lambda x:x[0] + x[1]*width)
        
        topLeft = points[0]
        topRight = points[1]
        bottomLeft = points[2]
        bottomRight = points[3]
        test_print(f'topLeft({type(topLeft)}):{topLeft} | ({type(bottomRight)}):{bottomRight} | ({type(topRight)}):{topRight} | ({type(bottomLeft)}):{bottomLeft}')
        points = [topLeft, topRight, bottomRight, bottomLeft]

        metadata = {"path": path, "carom_id":1, "TL":topLeft, "BR":bottomRight, "TR":topRight, "BL":bottomLeft, "WIDTH":width, "HIGHT":hight}
        images = {"origin":im0}
        input = PipeResource(im=im0, metadata=metadata, images=images, s=s)
        project_pipe.push_src(input)
        # 원본 정사영 영역 표시
        if display:
            origin = input.images["origin"].copy()
            for i in range(4):
                origin = cv2.line(origin, 
                        (int(points[i][0]), int(points[i][1])), 
                        (int(points[(i+1)%4][0]), int(points[(i+1)%4][1])), 
                        (0, 255, 0), 2)
            try:
                cv2.imshow("origin", origin)
                # input.imshow_table()
                input.imshow(name="proj", images="projected")
                cv2.waitKey(10000)
            except:pass
    bag_split.print()
    

def runner(args):
    print_args(vars(args))
    coord_test(args.src, not args.no_display)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default= (CAROM_BASE_DIR / "media" / "test2" / "sample.jpg"))
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--no_display', default=False, action="store_true")
    args = parser.parse_args()
    runner(args) 

  