from __future__ import annotations

from math import inf
import os
from collections import Counter
from typing import BinaryIO

import regex as re

# split on whitespace, letters, numbers, punctuation, and special characters
PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_tokens: list[bytes]) -> list[int]:
    # Copied from pretokenization_example.py
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = max(1, file_size // max(1, desired_num_chunks))
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[0] = 0
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    for idx in range(1, len(chunk_boundaries) - 1):
        start = chunk_boundaries[idx]
        file.seek(start)
        pos = start
        while True:
            block = file.read(mini_chunk_size)
            if block == b"":
                chunk_boundaries[idx] = file_size
                break
            mini_chunk_found_at = inf
            for split_special_token in split_special_tokens:
                found_at = block.find(split_special_token)
                if found_at != -1 and found_at < mini_chunk_found_at:
                    mini_chunk_found_at = found_at
            if mini_chunk_found_at != inf:
                chunk_boundaries[idx] = pos + int(mini_chunk_found_at)
                break
            pos += mini_chunk_size
    return sorted(set(chunk_boundaries))


def pretokenize_chunk(chunk_bytes: bytes, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    text = chunk_bytes.decode("utf-8", errors="ignore")
    if special_tokens:
        # necessary for regex to work
        escaped = [re.escape(token) for token in special_tokens]
        pieces = re.split("|".join(escaped), text)
    else:
        pieces = [text]
    counts: Counter[tuple[bytes, ...]] = Counter()
    for piece in pieces:
        if not piece.strip():
            continue
        for match in re.finditer(PATTERN, piece):
            token_bytes = match.group(0).encode("utf-8")
            counts[tuple(bytes([b]) for b in token_bytes)] += 1
    return counts

