import csv
from pathlib import Path

from ascon.state import random_state
from ascon.permutation import ascon_permutation_trace

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


def apply_input_difference(state):
    from ascon.state import AsconState

    return AsconState(
        state.x0 ^ 0x1,
        state.x1,
        state.x2,
        state.x3,
        state.x4,
    )


def generate_intermediate_dataset(rounds=4, target_round=1, samples=20000,
                                  output_file="results/intermediate_r4_round1.csv"):
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
            s2 = apply_input_difference(s)

            t1 = ascon_permutation_trace(s, rounds)
            t2 = ascon_permutation_trace(s2, rounds)

            dy = xor_states(t1[target_round - 1], t2[target_round - 1])
            writer.writerow(state_to_bits(dy) + [1])

        # Class 0: Random differential vectors
        for _ in range(samples - half):
            s1 = random_state()
            s2 = random_state()

            t1 = ascon_permutation_trace(s1, rounds)
            t2 = ascon_permutation_trace(s2, rounds)

            dy = xor_states(t1[target_round - 1], t2[target_round - 1])
            writer.writerow(state_to_bits(dy) + [0])

    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    generate_intermediate_dataset(rounds=4, target_round=1)