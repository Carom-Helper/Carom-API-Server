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
                           copy_piperesource, is_test, cv2, print_args)

def is_test_projection()->bool:
    return True and is_test()

def test_print(s, s1="", s2="", s3="", s4="", s5="", end="\n"):
    if is_test_projection():
        print("projection pipe test : ", s, s1, s2, s3, s4, s5, end=end)



class ProjectionPipe(One2OnePipe):
    def __init__(self):
        super().__init__()
        self.points = [[520,100],[970,102],[1440,650],[0,635]]

    @torch.no_grad()
    def exe(self, input: PipeResource) -> PipeResource:
        t1 = time.time()
        output = PipeResource()

        img = input.im.copy()

        pts = np.zeros((4, 2), dtype=np.float32)
        for i in range(4):
            pts[i] = self.points[i]
        
        sm = pts.sum(axis=1)  # 4쌍의 좌표 각각 x+y 계산
        diff = np.diff(pts, axis=1)  # 4쌍의 좌표 각각 x-y 계산

        topLeft = pts[np.argmin(sm)]  # x+y가 가장 값이 좌상단 좌표
        bottomRight = pts[np.argmax(sm)]  # x+y가 가장 큰 값이 우하단 좌표
        topRight = pts[np.argmin(diff)]  # x-y가 가장 작은 것이 우상단 좌표
        bottomLeft = pts[np.argmax(diff)]  # x-y가 가장 큰 값이 좌하단 좌표

        # 변환 전 4개 좌표 
        pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

        # 변환 후 4개 좌표
        pts2 = np.float32([[0, 0], [399, 0],
                                [399,799], [0, 799]])

        # 변환 행렬 계산 
        mtrx = cv2.getPerspectiveTransform(pts1, pts2)
        # 원근 변환 적용
        #result = cv2.warpPerspective(img, mtrx, (int(width), int(height)))
        result = cv2.warpPerspective(img, mtrx, (400, 800))
        t2 = time.time()

        # 원본 정사영 영역 표시
        origin = input.images["origin"].copy()
        for i in range(4):
            origin = cv2.line(origin, (pts[i][0], pts[i][1]), (pts[(i+1)%4][0], pts[(i+1)%4][1]), (0, 255, 0), 2)


        input.im = result.copy()
        input.images["projected"] = result
        output = input
        cv2.imshow("origin", origin)
        cv2.imshow("proj", output.im)

        return output

    def get_regist_type(self, idx=0) -> str:
        return "proj"

class ProjectionCoordPipe(One2OnePipe):
    def __init__(self, display=True):
        super().__init__()
        self.display = display
        #self.points = [[520,100],[970,102],[1440,650],[0,635]]
        self.points = [[549,109],[942,111],[1270,580],[180,565]]

    @torch.no_grad()
    def exe(self, input: PipeResource) -> PipeResource:
        t1 = time.time()

        img = input.im.copy()

        topLeft = input.metadata["TL"]  # x+y가 가장 값이 좌상단 좌표
        bottomRight = input.metadata["BR"]  # x+y가 가장 큰 값이 우하단 좌표
        topRight = input.metadata["TR"]  # x-y가 가장 작은 것이 우상단 좌표
        bottomLeft = input.metadata["BL"]   # x-y가 가장 큰 값이 좌하단 좌표

        # 변환 전 4개 좌표 
        pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

        # 변환 후 4개 좌표
        pts2 = np.float32([[0, 0], [399, 0],
                                [399,799], [0, 799]])

        # 변환 행렬 계산 
        mtrx = cv2.getPerspectiveTransform(pts1, pts2)

        # 원근 변환 적용
        #result = cv2.warpPerspective(img, mtrx, (int(width), int(height)))
        result = cv2.warpPerspective(img, mtrx, (400, 800))
        t2 = time.time()

        projected = np.zeros((800,400,1), np.uint8)
        for det in input.dets:
            #print(det["x"], det["y"])
            temp = np.array([det["x"] + det["w"]/2, det["y"] + det["h"]/2, 1])
            test_print(temp)
            temp_result = mtrx@temp
            projx = int(temp_result[0]/temp_result[2])
            projy = int(temp_result[1]/temp_result[2])
            test_print(temp_result[0]/temp_result[2], temp_result[1]/temp_result[2], "\n")
            result = cv2.line(result, (projx, projy), (projx, projy), (0,0,0), 5)
            projected = cv2.circle(projected, (projx, projy), 9, (255, 255, 255), 1)
        input.im = result.copy()
        input.images["projected"] = result
        output = input
        if self.display :
            cv2.imshow("proj", output.im)
            cv2.imshow("proj2", projected)

        return output

    def get_regist_type(self, idx=0) -> str:
        return "proj_coord"
    
    
def test(src, display=True):
    ### Pipe 생성###
    project_pipe = ProjectionCoordPipe(display=display)
    bag_split = ResourceBag()
    
    # 파이프 연결
    project_pipe.connect_pipe(bag_split)
    ### Dataloader ###
    dataset = LoadImages(src)
    ### 실행 ###
    for im0, path, s in dataset:
        #point 위치 확인
        points = [[549,109],[942,111],[1270,580],[180,565]]
        pts = np.zeros((4, 2), dtype=np.float32)
        for i in range(4):
            pts[i] = points[i]
        
        sm = pts.sum(axis=1)  # 4쌍의 좌표 각각 x+y 계산
        diff = np.diff(pts, axis=1)  # 4쌍의 좌표 각각 x-y 계산

        topLeft = pts[np.argmin(sm)]  # x+y가 가장 값이 좌상단 좌표
        bottomRight = pts[np.argmax(sm)]  # x+y가 가장 큰 값이 우하단 좌표
        topRight = pts[np.argmin(diff)]  # x-y가 가장 작은 것이 우상단 좌표
        bottomLeft = pts[np.argmax(diff)]  # x-y가 가장 큰 값이 좌하단 좌표
        test_print(f'topLeft({type(topLeft)}):{topLeft} | ({type(bottomRight)}):{bottomRight} | ({type(topRight)}):{topRight} | ({type(bottomLeft)}):{bottomLeft}')
        

        metadata = {"path": path, "TL":topLeft, "BR":bottomRight, "TR":topRight, "BL":bottomLeft}
        images = {"origin":im0}
        input = PipeResource(im=im0, metadata=metadata, images=images, s=s)
        project_pipe.push_src(input)
        # 원본 정사영 영역 표시
        if display:
            origin = input.images["origin"].copy()
            for i in range(4):
                origin = cv2.line(origin, (pts[i][0], pts[i][1]), (pts[(i+1)%4][0], pts[(i+1)%4][1]), (0, 255, 0), 2)
            cv2.imshow("origin", origin)
            cv2.waitKey(1000)
    bag_split.print()

def runner(args):
    print_args(vars(args))
    test(args.src, args.display)
    #test_singleton()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default= (CAROM_BASE_DIR / "media" / "test2" / "sample.jpg"))
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--display', default=True, action="store_false")
    args = parser.parse_args()
    runner(args) 

  