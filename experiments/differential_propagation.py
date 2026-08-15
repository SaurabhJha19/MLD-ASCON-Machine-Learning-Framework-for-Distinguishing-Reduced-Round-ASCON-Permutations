import random
import matplotlib.pyplot as plt

from ascon.state import random_state
from ascon.permutation import ascon_permutation_trace
from ascon.utils import hamming_weight

MASK64 = 0xFFFFFFFFFFFFFFFF


def apply_input_difference(state):
    s = state.copy()
    s.x0 ^= 0x1  # flip one bit in x0
    return s


def state_hamming_distance(a, b):
    dist = 0
    for x, y in zip(a.as_list(), b.as_list()):
        dist += hamming_weight((x ^ y) & MASK64)
    return dist


def average_propagation(rounds=4, samples=5000):
    totals = [0 for _ in range(rounds)]

    for _ in range(samples):
        s1 = random_state()
        s2 = apply_input_difference(s1)

        t1 = ascon_permutation_trace(s1, rounds)
        t2 = ascon_permutation_trace(s2, rounds)

        for r in range(rounds):
            totals[r] += state_hamming_distance(t1[r], t2[r])

    return [x / samples for x in totals]


if __name__ == "__main__":
    propagation = average_propagation(rounds=4, samples=5000)

    print("Average Hamming Distance per Round:")
    for i, v in enumerate(propagation, start=1):
        print(f"Round {i}: {v:.2f} bits")

    rounds = [1, 2, 3, 4]

    plt.figure(figsize=(6,4))
    plt.plot(rounds, propagation, marker='o', linewidth=2)
    plt.xticks(rounds)
    plt.xlabel('Round')
    plt.ylabel('Average Hamming Distance')
    plt.title('Differential Propagation in Reduced-Round ASCON')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('results/differential_propagation.png', dpi=300)
    print('Saved to results/differential_propagation.png')

    plt.show()