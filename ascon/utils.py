MASK64 = 0xFFFFFFFFFFFFFFFF


def rotr(x: int, n: int) -> int:
    #Rotate right
    x &= MASK64
    return ((x >> n) | (x << (64 - n))) & MASK64


def rotl(x: int, n: int) -> int:
    #Rotate left
    x &= MASK64
    return ((x << n) | (x >> (64 - n))) & MASK64


def xor64(a: int, b: int) -> int:
    return (a ^ b) & MASK64


def hamming_weight(x: int) -> int:
    return bin(x & MASK64).count("1")