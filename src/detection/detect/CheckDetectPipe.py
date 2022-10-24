from DetectError import *
from pipe_cls import *
import random


class CheckDetectPipe(One2OnePipe):
    def __init__(self) -> None:
        self.det_fail = 0
        super().__init__()

    def exe(self, input: PipeResource) -> PipeResource:
        if abs(len(input.dets)-3) != 0:
            self.det_fail += 1

        if self.det_fail >= 10:
            raise NotEnoughDetectError(
                "Too many errors have occured in Ball Detecting")

        return input

    def get_regist_type(self, idx=0) -> str:
        return "det_ball_check"


def test():
    pass_pipe = PassPipe()
    check_detect_pipe = CheckDetectPipe()
    pass_pipe.connect_pipe(check_detect_pipe)

    for i in range(10):
        num = random.randrange(1, 3)
        input = PipeResource()
        if num == 1:
            for i in range(2):
                input.append_det(xywh=[0, 0, 0, 0])
        else:
            for i in range(4):
                input.append_det(xywh=[0, 0, 0, 0])

        input.print()
        pass_pipe.push_src(input)


if __name__ == "__main__":
    test()
