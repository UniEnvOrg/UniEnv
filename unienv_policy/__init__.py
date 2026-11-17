"""Policy interfaces for UniEnv (merged from UniAct).

- ``Policy`` / ``PolicyWrapper``: stateful policies that observe an environment and compute actions.
- ``FuncPolicy`` / ``FuncPolicyWrapper``: functional policies whose state is passed in and returned explicitly.
- ``FuncPolicyBasedPolicy``: runs a ``FuncPolicy`` behind the stateful ``Policy`` interface.
"""
from .base import *
