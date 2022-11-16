from action_cls import *
    # IObserver, ISubject, IMoveable, ICrashable, is_test)

def is_test_wallobject()->bool:
    return False and is_test()

def test_print(s, s1="", s2="", s3="", s4="", s5="", end="\n"):
    if is_test_wallobject():
        print("wall object exe : ", s, s1, s2, s3, s4, s5, end=end)
        
        
class WallObject(IMovableObserver, CrashableSubject):
    def __init__(self) -> None:
        super().__init__()
        self.pos={"x":-1, "y":-1}
    
    
    