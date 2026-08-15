ROUND_CONSTANTS = [
    0xF0,
    0xE1,
    0xD2,
    0xC3,
    0xB4,
    0xA5,
    0x96,
    0x87,
    0x78,
    0x69,
    0x5A,
    0x4B,
]


def add_round_constant(state, rc):
    """Inject round constant into x2."""
    state.x2 ^= rc
    state.x2 &= 0xFFFFFFFFFFFFFFFF
    return state