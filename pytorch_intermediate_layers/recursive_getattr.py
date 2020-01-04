from typing import Any


def _recursive_getattr(obj: Any, name: str) -> Any:
    attr_name, *rest = name.split(".", maxsplit=1)
    if not _is_integer(attr_name):
        attr = getattr(obj, attr_name)
    else:
        attr = obj[int(attr_name)]
    if len(rest) > 0:
        return _recursive_getattr(attr, rest[0])
    else:
        return attr


def _is_integer(possible_int: str) -> bool:
    try:
        int(possible_int)
        return True
    except ValueError:
        return False
