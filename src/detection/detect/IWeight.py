from threading import Lock
from abc import *

class IWeight(metaclass=ABCMeta):
    @abstractmethod
    def inference(self, im, size):
        pass
    @abstractmethod
    def preprocess(self, im):
        pass