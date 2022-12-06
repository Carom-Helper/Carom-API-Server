import torch
import time

from pipe_cls import One2OnePipe, ResourceBag
from detect_utills import PipeResource

class CoordFilterPipe(One2OnePipe):
    def __init__(self) -> None:
        super().__init__()

    def exe(self, input: PipeResource) -> PipeResource:
        t1 = time.time()
        output = PipeResource()

        remove_list = []
        for det in input.dets:
            if not self.check(det):
                remove_list.append(det)
        for remove in remove_list:
            input.dets.remove(remove)

        if len(input.dets) > 3:
            input.dets = input.dets[:3]
        t2 = time.time()
        output = input
        return output
    
    def check(self, det: dict) -> bool:
        if det['x'] > 0 and det['x'] < 400 and det['y'] > 0 and det['y'] < 800:
            return True
        else:
            return False

    def get_regist_type(self, idx=0) -> str:
        return "coord_filter"