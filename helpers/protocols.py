from typing import Protocol


class HasLtMethod(Protocol):
    def __lt__(self, other) -> bool:
        raise NotImplementedError
