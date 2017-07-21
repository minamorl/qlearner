from enum import IntEnum


class Action(IntEnum):
    LONG = 0
    FLAT = 1
    SHORT = 2

    def __len__(self):
        return 3
