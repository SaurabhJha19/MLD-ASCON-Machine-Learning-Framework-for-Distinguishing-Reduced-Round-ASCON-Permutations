from ascon.state import AsconState

MASK64 = 0xFFFFFFFFFFFFFFFF


def sbox_layer(state: AsconState) -> AsconState:
    x0, x1, x2, x3, x4 = state.as_list()

    # Initial XOR layer
    x0 ^= x4
    x4 ^= x3
    x2 ^= x1

    # Nonlinear layer
    t0 = (~x0) & x1
    t1 = (~x1) & x2
    t2 = (~x2) & x3
    t3 = (~x3) & x4
    t4 = (~x4) & x0

    x0 ^= t1
    x1 ^= t2
    x2 ^= t3
    x3 ^= t4
    x4 ^= t0

    # Final XOR layer
    x1 ^= x0
    x0 ^= x4
    x3 ^= x2
    x2 = (~x2) & MASK64

    return AsconState(x0, x1, x2, x3, x4)