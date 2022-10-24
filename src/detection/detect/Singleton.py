from threading import Lock

class Singleton(type):
    _instance =None
    _lock = Lock()
    
    def __call__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instance