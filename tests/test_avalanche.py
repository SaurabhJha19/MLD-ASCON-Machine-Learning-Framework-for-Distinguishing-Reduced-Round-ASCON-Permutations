from ascon.state import random_state, AsconState
from ascon.permutation import ascon_p12
from ascon.utils import hamming_weight

MASK64 = 0xFFFFFFFFFFFFFFFF


def state_hamming_distance(a: AsconState, b: AsconState) -> int:
    dist = 0
    for x, y in zip(a.as_list(), b.as_list()):
        dist += hamming_weight((x ^ y) & MASK64)
    return dist


# Original random state
s1 = random_state()

# Copy and flip 1 bit
s2 = s1.copy()
s2.x0 ^= 1

# Apply full permutation
out1 = ascon_p12(s1)
out2 = ascon_p12(s2)

distance = state_hamming_distance(out1, out2)

print("Hamming distance:", distance)

# A good avalanche effect should change many bits
assert distance > 100

print("Avalanche test passed")