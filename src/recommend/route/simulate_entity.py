from abc import ABCMeta, abstractclassmethod
import numpy as np


class IMoveEntityHandleAble(metaclass=ABCMeta):
    @abstractclassmethod
    def set_next_position(vector:np.array)->float:
        """_summary_

        Args:
            vector (np.array): (x, y)

        Returns:
            float: moved distance
        """
        pass 
    @abstractclassmethod
    def set_next_speed(speed:float)->None:
        pass
    @abstractclassmethod
    def apply_move_action_result()->None:
        pass
    @abstractclassmethod
    def get_nomalized_vector()->np.array:
        pass
    @abstractclassmethod
    def get_speed()->float:
        pass
    @abstractclassmethod
    def get_upspin()->float:
        pass
    @abstractclassmethod
    def get_distance()->float:
        """_summary_

        Returns:
            float: if ball move, return distance
            or None: if ball don't move, return None
        """
        pass
    
class IActinoEntityHandleAble(metaclass=ABCMeta):
    @abstractclassmethod
    def rotate_next_vector(radian:float)->None:
        pass
    @abstractclassmethod
    def set_next_upspin(upspin:float)->None:
        pass
    @abstractclassmethod
    def set_next_sidespin(sidespin:float)->None:
        pass
    @abstractclassmethod
    def apply_crash_action_result()->None:
        pass
    @abstractclassmethod
    def get_upspin()->float:
        pass
    @abstractclassmethod
    def get_sidespin()->float:
        pass
    @abstractclassmethod
    def get_speed()->float:
        pass
    @abstractclassmethod
    def get_nomalized_vector()->np.array:
        pass
    @abstractclassmethod
    def get_xy()->list:
        """_summary_

        Returns:
            list: [x, y]
        """
        pass
   

    
class ICrashAble(metaclass=ABCMeta):
    @abstractclassmethod
    def change_your_ball_next_action_following_my_crash_policy(ball:IActinoEntityHandleAble):
        pass
    @abstractclassmethod
    def pass_vector_to_next_vector(vector:np.array)->None:
        pass
    @abstractclassmethod
    def get_nomal_vector(x,y)->np.array:
        pass
    @abstractclassmethod
    def get_volume_range()->float:
        pass
    # @abstractclassmethod
    # def get_closure_include_get_nomal_vector():
    #     pass
    

