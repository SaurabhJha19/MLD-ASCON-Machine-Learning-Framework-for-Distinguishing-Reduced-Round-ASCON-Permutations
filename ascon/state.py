from dataclasses import dataclass
import random

MASK64 = 0xFFFFFFFFFFFFFFFF


@dataclass
class AsconState:
    x0: int
    x1: int
    x2: int
    x3: int
    x4: int

    def __post_init__(self):
        self.x0 &= MASK64
        self.x1 &= MASK64
        self.x2 &= MASK64
        self.x3 &= MASK64
        self.x4 &= MASK64

    def copy(self):
        return AsconState(self.x0, self.x1, self.x2, self.x3, self.x4)

    def as_list(self):
        return [self.x0, self.x1, self.x2, self.x3, self.x4]

    def __repr__(self):
        return (
            f"AsconState(\\n"
            f"  x0=0x{self.x0:016x},\\n"
            f"  x1=0x{self.x1:016x},\\n"
            f"  x2=0x{self.x2:016x},\\n"
            f"  x3=0x{self.x3:016x},\\n"
            f"  x4=0x{self.x4:016x}\\n"
            f")"
        )


def random_state():
    return AsconState(
        random.getrandbits(64),
        random.getrandbits(64),
        random.getrandbits(64),
        random.getrandbits(64),
        random.getrandbits(64),
    )