import csv
from pathlib import Path

from ascon.state import random_state
from ascon.permutation import ascon_permutation

MASK64 = 0xFFFFFFFFFFFFFFFF


def state_to_bits(state):
    bits = []
    for word in state.as_list():
        for i in range(63, -1, -1):
            bits.append((word >> i) & 1)
    return bits


def xor_states(a, b):
    from ascon.state import AsconState

    return AsconState(
        (a.x0 ^ b.x0) & MASK64,
        (a.x1 ^ b.x1) & MASK64,
        (a.x2 ^ b.x2) & MASK64,
        (a.x3 ^ b.x3) & MASK64,
        (a.x4 ^ b.x4) & MASK64,
    )


def apply_input_difference(state, word, bit):
    from ascon.state import AsconState

    s = state.copy()
    mask = 1 << bit

    if word == 0:
        s.x0 ^= mask
    elif word == 1:
        s.x1 ^= mask
    elif word == 2:
        s.x2 ^= mask
    elif word == 3:
        s.x3 ^= mask
    elif word == 4:
        s.x4 ^= mask
    else:
        raise ValueError("word must be between 0 and 4")

    return s


def generate_dataset(rounds=4, samples=10000, output_file="results/differential_r4.csv", diff_word=0, diff_bit=0):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        header = [f"b{i}" for i in range(320)] + ["label"]
        writer.writerow(header)

        half = samples // 2

        # Class 1: True differential pairs
        for _ in range(half):
            s = random_state()
            s2 = apply_input_difference(s, diff_word, diff_bit)

            y1 = ascon_permutation(s, rounds)
            y2 = ascon_permutation(s2, rounds)

            dy = xor_states(y1, y2)

            writer.writerow(state_to_bits(dy) + [1])

        # Class 0: Random differential vectors
        for _ in range(samples - half):
            s1 = random_state()
            s2 = random_state()

            y1 = ascon_permutation(s1, rounds)
            y2 = ascon_permutation(s2, rounds)

            dy = xor_states(y1, y2)

            writer.writerow(state_to_bits(dy) + [0])

    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    generate_dataset(rounds=4, samples=10000, output_file="results/differential_r4.csv", diff_word=0, diff_bit=0)