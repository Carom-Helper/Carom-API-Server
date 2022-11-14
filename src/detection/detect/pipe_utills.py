

# set path
import sys
from pathlib import Path
import os


CAROM_BASE_DIR=Path(__file__).resolve().parent.parent.parent
FILE = Path(__file__).resolve()
ROOT = FILE.parent

tmp = CAROM_BASE_DIR
if str(tmp) not in sys.path and os.path.isabs(tmp):
    sys.path.append(str(tmp))  # add ROOT to PATH


from detect_utills import PipeResource, copy_piperesource, is_test
from pipe_cls import IPipeObserver, One2OnePipe

def is_test_pipe_utills()->bool:
    return True and is_test()

def test_print(s, s1="", s2="", s3="", s4="", s5="", end="\n"):
    if is_test_pipe_utills():
        print("pipe cls exe : ", s, s1, s2, s3, s4, s5, end=end)
        
        
class SaveBallCoordPipe(IPipeObserver):
    def __init__(self, display=False) -> None:
        super().__init__()
        self.src=None
        self.display = display
    
    def push_src(self, input: PipeResource) -> None:
        from detection.models import balls_coord
        self.src = input
        
        carom_id = input.metadata["carom_id"]
        coord = dict()
        for i, det in enumerate(input):
            coord[str(i+1)] = [int(det["x"]), int(det["y"])]
        test_print("coord : ", coord)
        ball = balls_coord(carom_id=carom_id, coord=coord)
        ball.save()
            
    def print(self):
        if self.src is not None:
            self.src.print()
    
    
