from threading import Lock

class PIPE_Singleton(type):
    _instance =None
    _lock = Lock()
    
    def __call__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(PIPE_Singleton, cls).__call__(*args, **kwargs)
        return cls._instance


class GPU_YOLO_Singleton(type):
    _instance =None
    _lock = Lock()
    
    def __call__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(GPU_YOLO_Singleton, cls).__call__(*args, **kwargs)
        return cls._instance
    
class NPU_YOLO_Singleton(type):
    _instance =None
    _lock = Lock()
    
    def __call__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(NPU_YOLO_Singleton, cls).__call__(*args, **kwargs)
        return cls._instance