import torch
import time
from random import randint

from pipe_cls import One2OnePipe, ResourceBag
from detect_utills import PipeResource, is_test

def is_test_factory()->bool:
    return True and is_test()
class BallGeneratePipe(One2OnePipe):
    def __init__(self) -> None:
        super().__init__()
    
    def exe(self, input: PipeResource) -> PipeResource:
        t1 = time.time()
        output = PipeResource()
        while len(input.dets) < 3:
            new_ball = self.generate()
            check = True
            for ball in input.dets:
                if not self.check_dist(ball, new_ball):
                    check = False
                    break
            if check:
                input.dets.append(new_ball)

        t2 = time.time()
        output = input
        if is_test_factory():
            output.print()
        return output
    
    def generate(self) -> dict:
        new_ball = {'xmin': 0, 'ymin': 0, 'xmax': 0, 'ymax': 0, 'conf': 0, 'cls': 1, 'label': 'BALL', 'x': 0, 'y': 780, 'w': 0, 'h': 0}
        x = randint(20, 380)
        new_ball['x'] = x
        return new_ball

    def check_dist(self, b1, b2) -> bool:
        dist = ((b1['x']-b2['x'])**2 + (b1['y'] - b2['y'])**2)**0.5
        return True if dist >= 10 else False


    def get_regist_type(self, idx=0) -> str:
        return "ball_genereate"