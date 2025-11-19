import os
import json
from typing import Iterable, Iterator
import regex as re
import numpy as np
import time
import tqdm

PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        self.bytes_to_token = {v: k for k, v in self.vocab.items()}
        if special_tokens:
            special_tokens = sorted(special_tokens, key=len, reverse=True)
            self.special_tokens = special_tokens
            self.special_tokens_set = set(special_tokens)
            self.special_tokens_pattern = "|".join([re.escape(token) for token in special_tokens])
            for special_token in special_tokens:
                byte_encoded = special_token.encode("utf-8")
                if byte_encoded not in self.bytes_to_token:
                    self.vocab[len(self.vocab)] = byte_encoded
                    self.bytes_to_token[byte_encoded] = len(self.vocab) - 1

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        from tests.common import gpt2_bytes_to_unicode
        byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath, "r") as f:
            vocab_dict = json.load(f)
        vocab = {}
        for k, v in vocab_dict.items():
            if isinstance(v, str):
                token_bytes = bytes([byte_decoder[char] for char in v])
            else:
                token_bytes = bytes(v) if isinstance(v, list) else v
            vocab[int(k)] = token_bytes
        with open(merges_filepath, "r") as f:
            merges = []
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    left_bytes = bytes([byte_decoder[char] for char in parts[0]])
                    right_bytes = bytes([byte_decoder[char] for char in parts[1]])
                    merges.append((left_bytes, right_bytes))
        return cls(vocab, merges, special_tokens)

    def _apply_merges(self, pre_token_bytes: list[bytes]) -> list[bytes]:
        tokens = pre_token_bytes.copy()
        for merge_token_1, merge_token_2 in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == merge_token_1 and tokens[i + 1] == merge_token_2:
                    new_tokens.append(merge_token_1 + merge_token_2)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text: str, verbose: bool = False) -> list[int]:
        if not text:
            return []
        
        token_ids = []
        if self.special_tokens:
            split_parts = re.split(f"({self.special_tokens_pattern})", text)
        else:
            split_parts = [text]
        
        if verbose:
            progress_bar = tqdm.tqdm(total=len(split_parts), desc="Encoding text")
        for part in split_parts:
            if verbose:
                progress_bar.update(1)
            if not part:
                continue

            if self.special_tokens and part in self.special_tokens_set:
                special_bytes = part.encode("utf-8")
                if special_bytes in self.bytes_to_token:
                    token_ids.append(self.bytes_to_token[special_bytes])
                continue
            
            for match in re.finditer(PATTERN, part):
                pre_token = match.group(0)
                pre_token_bytes = pre_token.encode("utf-8")
                byte_list = [bytes([b]) for b in pre_token_bytes]
                merged_tokens = self._apply_merges(byte_list)
                for token in merged_tokens:
                    if token in self.bytes_to_token:
                        token_ids.append(self.bytes_to_token[token])
                    else:
                        for b in token:
                            single_byte = bytes([b])
                            if single_byte in self.bytes_to_token:
                                token_ids.append(self.bytes_to_token[single_byte])
        
        if verbose:
            progress_bar.close()
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        accumulated = ""
        for text in iterable:
            accumulated += text
            if self.special_tokens:
                found = False
                for special_token in self.special_tokens:
                    if special_token in accumulated:
                        idx = accumulated.find(special_token)
                        if idx > 0:
                            for token_id in self.encode(accumulated[:idx]):
                                yield token_id
                        special_bytes = special_token.encode("utf-8")
                        if special_bytes in self.bytes_to_token:
                            yield self.bytes_to_token[special_bytes]
                        accumulated = accumulated[idx + len(special_token):]
                        found = True
                        break
                if not found:
                    continue
            else:
                for token_id in self.encode(accumulated):
                    yield token_id
                accumulated = ""
        if accumulated:
            for token_id in self.encode(accumulated):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        text_bytes = b"".join([self.vocab[id] for id in ids])
        return text_bytes.decode("utf-8", errors="replace")

def test_maximum_speed():
    tokenizer = Tokenizer.from_files(
        vocab_filepath="outputs/tokenizer/tinystories_10k/vocab.json",
        merges_filepath="outputs/tokenizer/tinystories_10k/merges.txt",
        special_tokens=["<|endoftext|>"],
    )
    with open("data/TinyStoriesV2-GPT4-valid.txt", "r") as f:
        data = f.read()
    start_time = time.perf_counter()
    ids = tokenizer.encode(data[:10000], verbose=True)
    end_time = time.perf_counter()
    num_bytes = data[:10000].encode("utf-8")
    print(f"Time taken to encode validation dataset: {end_time - start_time} seconds")
    print(f"Speed for validation dataset: {len(num_bytes) / (end_time - start_time)} bytes/second")

def encode_tiny_stories(num_processes: int):
    tokenizer = Tokenizer.from_files(
        vocab_filepath="outputs/tokenizer/tinystories_10k/vocab.json",
        merges_filepath="outputs/tokenizer/tinystories_10k/merges.txt",
        special_tokens=["<|endoftext|>"],
    )
    from multiprocessing import Pool
    from basics.bpe_helpers import find_chunk_boundaries

    analysis = {}

    print("Encoding validation dataset...")
    print("Reading validation dataset...")
    with open("data/TinyStoriesV2-GPT4-valid.txt", "r") as f:
        data = f.read()
    print(f"Data size: {len(data)} bytes")
    start_time = time.perf_counter()
    chunks = find_chunk_boundaries(open("data/TinyStoriesV2-GPT4-valid.txt", "rb"), num_processes, [b"<|endoftext|>"])
    with Pool(processes=num_processes) as pool:
        val_ids = pool.map(tokenizer.encode, [data[chunks[i]:chunks[i+1]] for i in range(len(chunks) - 1)])
    final_val_ids = []
    for ids in val_ids:
        final_val_ids.extend(ids)
    val_ids = np.array(final_val_ids, dtype=np.uint16)
    with open("outputs/tokenizer/tinystories_10k/valid.ids", "wb") as f:
        val_ids.tofile(f)
    end_time = time.perf_counter()
    print(f"Time taken to encode validation dataset: {end_time - start_time} seconds")
    print(f"Validation dataset size: {len(val_ids)} tokens")
    print(f"Compression ratio for validation dataset: {len(data) / len(val_ids)} bytes/token")
    print(f"Speed for validation dataset: {len(data) / (end_time - start_time) / (max(1, num_processes))} bytes/second")
    analysis["validation"] = {
        "time": end_time - start_time,
        "data_size": len(data),
        "token_size": len(final_val_ids),
        "compression_ratio": len(data) / len(val_ids),
        "speed": len(data) / (end_time - start_time) / (max(1, num_processes)),
    }

    print("Encoding training dataset...")
    print("Reading training dataset...")
    with open("data/TinyStoriesV2-GPT4-train.txt", "r") as f:
        data = f.read()
    print(f"Data size: {len(data)} bytes")
    start_time = time.perf_counter()
    chunks = find_chunk_boundaries(open("data/TinyStoriesV2-GPT4-train.txt", "rb"), num_processes, [b"<|endoftext|>"])
    with Pool(processes=num_processes) as pool:
        train_ids = pool.map(tokenizer.encode, [data[chunks[i]:chunks[i+1]] for i in range(len(chunks) - 1)])
    final_train_ids = []
    for ids in train_ids:
        final_train_ids.extend(ids)
    train_ids = np.array(final_train_ids, dtype=np.uint16)
    with open("outputs/tokenizer/tinystories_10k/train.ids", "wb") as f:
        train_ids.tofile(f)
    end_time = time.perf_counter()
    print(f"Time taken to encode training dataset: {end_time - start_time} seconds")
    print(f"Training dataset size: {len(train_ids)} tokens")
    print(f"Compression ratio for training dataset: {len(data) / len(train_ids)} bytes/token")
    print(f"Speed for training dataset: {len(data) / (end_time - start_time) / (max(1, num_processes))} bytes/second")
    analysis["training"] = {
        "time": end_time - start_time,
        "data_size": len(data),
        "token_size": len(final_train_ids),
        "compression_ratio": len(data) / len(train_ids),
        "speed": len(data) / (end_time - start_time) / (max(1, num_processes)),
    }

    with open("outputs/tokenizer/tinystories_10k/analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis saved to outputs/tokenizer/tinystories_10k/analysis.json")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=16)
    args = parser.parse_args()
    test_maximum_speed()
    encode_tiny_stories(args.num_processes)
