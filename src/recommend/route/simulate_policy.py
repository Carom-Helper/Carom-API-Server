from abc import ABCMeta, abstractclassmethod
import simulate_entity as Entity

class MOVE_POLICY(metaclass=ABCMeta):
    def __call__(self, object:Entity.IMoveEntityHandleAble, time:float) -> None:
        return self.action(object, time)

    @abstractclassmethod
    def action(self, object:Entity.IMoveEntityHandleAble, time:float) -> None:
        pass

class CHANGE_SPEED_POLICY(metaclass=ABCMeta):
    def __call__(self, object:Entity.IMoveEntityHandleAble, distance:float):
        return self.action(object, distance)

    @abstractclassmethod
    def action(self, object:Entity.IMoveEntityHandleAble, distance:float):
        pass
    
class CRASH_POLICY(metaclass=ABCMeta):
    def __call__(self, main_obj: Entity.IActinoEntityHandleAble, crashable_obj:Entity.ICrashAble) -> bool:
        return self.action(main_obj, crashable_obj)

    @abstractclassmethod
    def action(self, main_obj: Entity.IActinoEntityHandleAble, crashable_obj:Entity.ICrashAble):
        pass
 
class CHECK_CRASH_POLICY(metaclass=ABCMeta):
    def __call__(self, obj1:Entity.ICrashAble, obj2:Entity.ICrashAble) -> bool:
        return self.check_crash(obj1, obj2)
    @abstractclassmethod
    def check_crash(self, obj1:Entity.ICrashAble, obj2:Entity.ICrashAble)->bool:
        pass

class IMoveActionHandler(metaclass=ABCMeta):
    @abstractclassmethod
    def move_action(time)->None:
        pass
    @abstractclassmethod
    def add_change_policy(chage_policy:CHANGE_SPEED_POLICY)->None:
        pass
    @abstractclassmethod
    def add_move_policy(move_policy:MOVE_POLICY)->None:
        pass
    @abstractclassmethod
    def init_move_policy()->None:
        pass
    @abstractclassmethod
    def init_change_policy()->None:
        pass
   
class ICrashActionHandler(metaclass=ABCMeta):
    @abstractclassmethod
    def crash_action(ball: Entity.IActinoEntityHandleAble)->None:
        pass
    @abstractclassmethod
    def add_crash_policy(crash_policy:CRASH_POLICY):
        pass
    @abstractclassmethod
    def init_crash_policy():
        pass