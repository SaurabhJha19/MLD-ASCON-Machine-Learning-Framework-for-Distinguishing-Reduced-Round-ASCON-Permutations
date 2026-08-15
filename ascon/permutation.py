from ascon.round import ascon_round
from ascon.constants import ROUND_CONSTANTS


def ascon_permutation(state, rounds=12):
    """
    Apply the ASCON permutation for the specified number of rounds.
    rounds can be 2, 3, 4, 5, 6, 8, or 12.
    """
    if rounds < 1 or rounds > 12:
        raise ValueError("Rounds must be between 1 and 12")

    # Use the last 'rounds' constants, matching the ASCON specification
    constants = ROUND_CONSTANTS[-rounds:]

    s = state.copy()
    for rc in constants:
        s = ascon_round(s, rc)

    return s


def ascon_p12(state):
    return ascon_permutation(state, 12)


def ascon_p8(state):
    return ascon_permutation(state, 8)


def ascon_p6(state):
    return ascon_permutation(state, 6)


def ascon_p5(state):
    return ascon_permutation(state, 5)


def ascon_p4(state):
    return ascon_permutation(state, 4)


def ascon_p3(state):
    return ascon_permutation(state, 3)


def ascon_p2(state):
    return ascon_permutation(state, 2)