import dataclasses
from typing import TypeVar
import functools

__all__ = ['stateclass', 'field', 'StateClass']

def field(pytree_node=True, *, metadata=None, **kwargs):
    return dataclasses.field(metadata=(metadata or {}) | {'pytree_node': pytree_node},
                        **kwargs)

def stateclass(
    clz=None, /, **kwargs
):
    if clz is None:
        return functools.partial(stateclass, **kwargs)  # type: ignore[bad-return-type]

    # check if already a stateclass
    if '_unienv_stateclass' in clz.__dict__:
        return clz

    if 'frozen' not in kwargs.keys():
        kwargs['frozen'] = True
    data_clz = dataclasses.dataclass(**kwargs)(clz)  # type: ignore
    
    def replace(self, **updates):
        """Returns a new object replacing the specified fields with new values."""
        return dataclasses.replace(self, **updates)

    data_clz.replace = replace

    # add a _unienv_stateclass flag to distinguish from regular dataclasses
    data_clz._unienv_stateclass = True  # type: ignore[attr-defined]

    return data_clz  # type: ignore

TNode = TypeVar('TNode', bound='StateClass')

class StateClass:
	def __init_subclass__(cls, **kwargs):
		stateclass(cls, **kwargs)  # pytype: disable=wrong-arg-types

	def __init__(self, *args, **kwargs):
		raise NotImplementedError
    
	def replace(self: TNode, **overrides) -> TNode:
		raise NotImplementedError
