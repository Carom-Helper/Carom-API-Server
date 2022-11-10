from pipe_cls import *
import torch
import numpy as np

class ProjectionPipe(One2OnePipe):
    def __init__(self):
        super().__init__()
        self.points = [[520,100],[970,102],[1440,650],[0,635]]

    @torch.no_grad()
    def exe(self, input: PipeResource) -> PipeResource:
        t1 = time.time()
        output = PipeResource()

        img = input.im0s.copy()

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
        origin = input.im0s.copy()
        for i in range(4):
            origin = cv2.line(origin, (pts[i][0], pts[i][1]), (pts[(i+1)%4][0], pts[(i+1)%4][1]), (0, 255, 0), 2)


        input.im = result.copy()
        output = copy_piperesource(input)
        output.im0s = output.im.copy()
        cv2.imshow("origin", origin)
        cv2.imshow("proj", output.im)
        cv2.waitKey(0)

        return output

    def get_regist_type(self, idx=0) -> str:
        return "proj"

class ProjectionCoordPipe(One2OnePipe):
    def __init__(self):
        super().__init__()
        #self.points = [[520,100],[970,102],[1440,650],[0,635]]
        self.points = [[549,109],[942,111],[1270,580],[180,565]]

    @torch.no_grad()
    def exe(self, input: PipeResource) -> PipeResource:
        t1 = time.time()
        output = PipeResource()

        img = input.im0s.copy()

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

        projected = np.zeros((800,400,1), np.uint8)
        for det in input.dets:
            #print(det["x"], det["y"])
            temp = np.array([det["x"] + det["w"]/2, det["y"] + det["h"]/2, 1])
            print(temp)
            temp_result = mtrx@temp
            projx = int(temp_result[0]/temp_result[2])
            projy = int(temp_result[1]/temp_result[2])
            print(temp_result[0]/temp_result[2], temp_result[1]/temp_result[2], "\n")
            result = cv2.line(result, (projx, projy), (projx, projy), (0,0,0), 5)
            projected = cv2.circle(projected, (projx, projy), 9, (255, 255, 255), 1)

        # 원본 정사영 영역 표시
        origin = input.im0s.copy()
        for i in range(4):
            origin = cv2.line(origin, (pts[i][0], pts[i][1]), (pts[(i+1)%4][0], pts[(i+1)%4][1]), (0, 255, 0), 2)
        



        input.im = result.copy()
        output = copy_piperesource(input)
        output.im0s = output.im.copy()
        cv2.imshow("origin", origin)
        cv2.imshow("proj", output.im)
        cv2.imshow("proj2", projected)
        cv2.waitKey(0)

        return output

    def get_regist_type(self, idx=0) -> str:
        return "proj_coord"