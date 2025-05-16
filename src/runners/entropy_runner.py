import math
from typing import List


def shannon(counts: List[int]) -> float:
    n = sum(counts)
    return -sum([(c / n) * math.log(c / n, 2) for c in counts if c > 0])


def main():
    cases = [
        [5000] + [1000 for _ in range(5)],
        [1000 for _ in range(10)],
        [50 for _ in range(200)],
        [5000] + [50 for _ in range(100)]
    ]

    for case in cases:
        print(shannon(case))

if __name__ == '__main__':
    main()