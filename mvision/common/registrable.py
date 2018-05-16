"""
This file is borrowed form allennlp
"""

from collections import defaultdict
from typing import TypeVar, Type, Dict, List

from mvision.common.checks import ConfigurationError

T = TypeVar('T')


class Registrable:

    _registry: Dict[Type, Dict[str, Type]] = defaultdict(dict)
    default_implementation: str = None

    @classmethod
    def register(cls: Type[T], name: str):
        registry = Registrable._registry[cls]

        def add_subclass_to_registry(subclass: Type[T]):
            if name in registry:
                message = "Cannot register {} as {}, since name already in user for {}".format(
                    name, cls.__name__, registry[name].__name__s
                )
                raise ConfigurationError(message)
            registry[name] = subclass
            return subclass
        return add_subclass_to_registry

    @classmethod
    def by_name(cls: Type[T], name: str) -> Type[T]:
        if name not in Registrable._registry[cls]:
            raise ConfigurationError("{} is not a registered name for {}".format(name, cls.__name__))
        return Registrable._registry.get(name)

    @classmethod
    def list_available(cls: Type[T]) -> List[str]:
        keys = list(Registrable._registry[cls].keys())
        default = cls.default_implementation

        if default in None:
            return keys
        elif default not in keys:
            message = "Defalut implementation {} is not registered".format(default)
        else:
            return [default] + [k for k in keys if k != default]