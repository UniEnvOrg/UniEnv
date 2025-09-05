import cloudpickle
from typing import Callable

class MultiProcessFn:
    def __init__(self, fn : Callable):
        self.fn = fn
    
    def __getstate__(self):
        return cloudpickle.dumps(self.fn)
    
    def __setstate__(self, state):
        self.fn = cloudpickle.loads(state)
    
    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)