from __future__ import annotations

NUM_JOINTS = 25

# Placeholder Body25-like topology. Verify the exact source mapping before using
# this for quantitative analysis; it is intentionally easy to edit.
BONES_25: list[tuple[int, int]] = [
    (8, 1),
    (1, 0),
    (0, 15),
    (15, 17),
    (0, 16),
    (16, 18),
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 22),
    (11, 24),
    (22, 23),
    (8, 12),
    (12, 13),
    (13, 14),
    (14, 19),
    (14, 21),
    (19, 20),
]
