from ascon.round import ascon_round
from ascon.constants import ROUND_CONSTANTS


def ascon_permutation(state, rounds=12):
    if rounds < 1 or rounds > 12:
        raise ValueError("Rounds must be between 1 and 12")

    constants = ROUND_CONSTANTS[-rounds:]
    s = state.copy()

    for rc in constants:
        s = ascon_round(s, rc)

    return s


def ascon_permutation_trace(state, rounds=12):
    """
    Return the intermediate state after every round.
    states[0] = after round 1
    states[1] = after round 2
    ...
    states[-1] = after final round
    """
    if rounds < 1 or rounds > 12:
        raise ValueError("Rounds must be between 1 and 12")

    constants = ROUND_CONSTANTS[-rounds:]
    s = state.copy()
    trace = []

    for rc in constants:
        s = ascon_round(s, rc)
        trace.append(s.copy())

    return trace


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