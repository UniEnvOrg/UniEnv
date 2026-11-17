"""Policy interfaces.

- ``Policy`` / ``PolicyWrapper``: stateful policies that observe an environment and compute actions.
- ``FuncPolicy`` / ``FuncPolicyWrapper``: functional policies whose state is passed in and returned explicitly.
- ``FuncPolicyBasedPolicy``: runs a ``FuncPolicy`` behind the stateful ``Policy`` interface.
"""
from .policy import Policy, PolicyWrapper
from .funcpolicy import FuncPolicy, FuncPolicyWrapper, FuncPolicyBasedPolicy

__all__ = ["Policy", "PolicyWrapper", "FuncPolicy", "FuncPolicyWrapper", "FuncPolicyBasedPolicy"]
