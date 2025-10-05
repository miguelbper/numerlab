from collections.abc import Callable
from typing import Any


def param_namer(fixture_name: str) -> Callable[[Any], str]:
    def namer(param: Any) -> str:
        return f"<{fixture_name}={param}>"

    return namer
