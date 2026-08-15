from ascon.state import AsconState
from ascon.utils import rotr


def diffusion_layer(state: AsconState) -> AsconState:
    x0, x1, x2, x3, x4 = state.as_list()

    x0 ^= rotr(x0, 19) ^ rotr(x0, 28)
    x1 ^= rotr(x1, 61) ^ rotr(x1, 39)
    x2 ^= rotr(x2, 1) ^ rotr(x2, 6)
    x3 ^= rotr(x3, 10) ^ rotr(x3, 17)
    x4 ^= rotr(x4, 7) ^ rotr(x4, 41)

    return AsconState(x0, x1, x2, x3, x4)