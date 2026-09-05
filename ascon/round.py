from ascon.constants import add_round_constant
from ascon.sbox import sbox_layer
from ascon.diffusion import diffusion_layer


def ascon_round(state, round_constant):
    state = add_round_constant(state.copy(), round_constant)
    state = sbox_layer(state)
    state = diffusion_layer(state)
    return state