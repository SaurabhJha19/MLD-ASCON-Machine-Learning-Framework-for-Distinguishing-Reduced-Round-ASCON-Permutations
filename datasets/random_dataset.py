import random
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


def random_bits_320():
    return [random.randint(0, 1) for _ in range(320)]


def generate_dataset(rounds=4, samples=10000, output_file="results/random_vs_ascon_r4.csv"):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        header = [f"b{i}" for i in range(320)] + ["label"]
        writer.writerow(header)

        half = samples // 2

        #True ASCON outputs
        for _ in range(half):
            s = random_state()
            out = ascon_permutation(s, rounds)
            writer.writerow(state_to_bits(out) + [1])

        #Random 320-bit vectors
        for _ in range(samples - half):
            writer.writerow(random_bits_320() + [0])

    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    generate_dataset(rounds=4, samples=10000)