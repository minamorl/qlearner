from enum import Enum

class Action(Enum):
    LONG = 0
    FLAT = 1
    SHORT = 1

    def __len__(self):
        return 3