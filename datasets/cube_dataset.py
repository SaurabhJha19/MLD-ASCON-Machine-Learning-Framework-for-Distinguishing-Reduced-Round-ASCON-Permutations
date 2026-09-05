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


def cube_sum(base_state, rounds=4):
    acc = AsconState(0, 0, 0, 0, 0)

    for v in range(256):
        s = base_state.copy()
        s.x0 = (s.x0 & 0xFFFFFFFFFFFFFF00) | v

        out = ascon_permutation(s, rounds)

        acc = xor_states(acc, out)

    return acc


def random_cube_sum(rounds=4):

    acc = AsconState(0, 0, 0, 0, 0)

    for _ in range(256):
        s = random_state()
        out = ascon_permutation(s, rounds)
        acc = xor_states(acc, out)

    return acc


def generate_dataset(rounds=4, samples=5000, output_file="results/cube_r4.csv"):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        header = [f"b{i}" for i in range(320)] + ["label"]
        writer.writerow(header)

        half = samples // 2

        #True cube sums
        for _ in range(half):
            base = random_state()
            vec = cube_sum(base, rounds)
            writer.writerow(state_to_bits(vec) + [1])

        #Random cube sums
        for _ in range(samples - half):
            vec = random_cube_sum(rounds)
            writer.writerow(state_to_bits(vec) + [0])

    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    generate_dataset(rounds=4, samples=5000)