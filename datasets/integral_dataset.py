import csv
from pathlib import Path

from ascon.state import random_state, AsconState
from ascon.permutation import ascon_permutation

MASK64 = 0xFFFFFFFFFFFFFFFF


def state_to_bits(state):
    bits = []
    for word in state.as_list():
        for i in range(63, -1, -1):
            bits.append((word >> i) & 1)
    return bits


def xor_states(a, b):
    return AsconState(
        (a.x0 ^ b.x0) & MASK64,
        (a.x1 ^ b.x1) & MASK64,
        (a.x2 ^ b.x2) & MASK64,
        (a.x3 ^ b.x3) & MASK64,
        (a.x4 ^ b.x4) & MASK64,
    )


def integral_vector(base_state, rounds=4, active_word=0):
    """
    Build an integral by varying the lowest 8 bits of one ASCON word.
    active_word: 0..4 corresponding to x0..x4
    """
    acc = AsconState(0, 0, 0, 0, 0)

    for v in range(256):
        s = base_state.copy()

        if active_word == 0:
            s.x0 = (s.x0 & 0xFFFFFFFFFFFFFF00) | v
        elif active_word == 1:
            s.x1 = (s.x1 & 0xFFFFFFFFFFFFFF00) | v
        elif active_word == 2:
            s.x2 = (s.x2 & 0xFFFFFFFFFFFFFF00) | v
        elif active_word == 3:
            s.x3 = (s.x3 & 0xFFFFFFFFFFFFFF00) | v
        elif active_word == 4:
            s.x4 = (s.x4 & 0xFFFFFFFFFFFFFF00) | v

        out = ascon_permutation(s, rounds)
        acc = xor_states(acc, out)

    return acc


def generate_dataset(rounds=4, samples=5000, active_word=0, output_file="results/integral_r4.csv"):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        header = [f"b{i}" for i in range(320)] + ["label"]
        writer.writerow(header)

        half = samples // 2

        # Class 1: True integral vectors
        for _ in range(half):
            base = random_state()
            vec = integral_vector(base, rounds, active_word)
            writer.writerow(state_to_bits(vec) + [1])

        # Class 0: Random XOR aggregates
        for _ in range(samples - half):
            acc = AsconState(0, 0, 0, 0, 0)

            for _ in range(256):
                s = random_state()
                out = ascon_permutation(s, rounds)
                acc = xor_states(acc, out)

            writer.writerow(state_to_bits(acc) + [0])

    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    generate_dataset(rounds=4, samples=5000, active_word=0,)