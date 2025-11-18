from __future__ import annotations

import argparse
import cProfile
import json
import time
from pathlib import Path
from pstats import Stats

from collections import Counter
from multiprocessing import Pool, cpu_count

import psutil
import tqdm

from tests.common import gpt2_bytes_to_unicode
from basics.bpe_helpers import find_chunk_boundaries, pretokenize_chunk


def run_train_bpe_local(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]], dict[str, float]]:
    timings: dict[str, float] = {}
    phase_start = time.perf_counter()

    vocab: dict[int, bytes] = {}
    token_id = 0
    for special_token in special_tokens:
        vocab[token_id] = special_token.encode("utf-8")
        token_id += 1
    for byte_val in range(256):
        vocab[token_id] = bytes([byte_val])
        token_id += 1

    timings["setup_seconds"] = time.perf_counter() - phase_start
    phase_start = time.perf_counter()

    num_processes = min(cpu_count(), 16)
    split_tokens = [token.encode("utf-8") for token in special_tokens]
    pre_token_counts: Counter[tuple[bytes, ...]] = Counter()
    chunk_args: list[tuple[bytes, list[str]]] = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_args.append((f.read(end - start), special_tokens))
    if chunk_args:
        workers = max(1, min(num_processes, len(chunk_args)))
        if workers == 1:
            for chunk_bytes, tokens in chunk_args:
                pre_token_counts.update(pretokenize_chunk(chunk_bytes, tokens))
        else:
            with Pool(processes=workers) as pool:
                for result in pool.starmap(pretokenize_chunk, chunk_args):
                    pre_token_counts.update(result)
    timings["pretokenization_seconds"] = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    
    merges: list[tuple[bytes, bytes]] = []
    pre_token_list: list[tuple[list[bytes], int]] = [
        (list(byte_tuple), count) for byte_tuple, count in pre_token_counts.items()
    ]
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    for pre_token_bytes, count in pre_token_list:
        for i in range(len(pre_token_bytes) - 1):
            pair = (pre_token_bytes[i], pre_token_bytes[i + 1])
            pair_counts[pair] += count
    num_merges_needed = vocab_size - len(vocab)
    for _ in tqdm.trange(num_merges_needed):
        if not pair_counts:
            break
        max_count = max(pair_counts.values())
        candidates = [pair for pair, count in pair_counts.items() if count == max_count]
        best_pair = max(candidates)
        merge_token_1, merge_token_2 = best_pair
        merged_token = merge_token_1 + merge_token_2
        vocab[token_id] = merged_token
        token_id += 1
        merges.append((merge_token_1, merge_token_2))
        new_pre_token_list: list[tuple[list[bytes], int]] = []
        for pre_token_bytes, count in pre_token_list:
            has_pair = False
            for i in range(len(pre_token_bytes) - 1):
                if pre_token_bytes[i] == merge_token_1 and pre_token_bytes[i + 1] == merge_token_2:
                    has_pair = True
                    break
            if not has_pair:
                new_pre_token_list.append((pre_token_bytes, count))
                continue
            for i in range(len(pre_token_bytes) - 1):
                old_pair = (pre_token_bytes[i], pre_token_bytes[i + 1])
                pair_counts[old_pair] -= count
                if pair_counts[old_pair] <= 0:
                    del pair_counts[old_pair]
            new_pre_token: list[bytes] = []
            i = 0
            while i < len(pre_token_bytes):
                if (
                    i < len(pre_token_bytes) - 1
                    and pre_token_bytes[i] == merge_token_1
                    and pre_token_bytes[i + 1] == merge_token_2
                ):
                    new_pre_token.append(merged_token)
                    i += 2
                else:
                    new_pre_token.append(pre_token_bytes[i])
                    i += 1
            for i in range(len(new_pre_token) - 1):
                new_pair = (new_pre_token[i], new_pre_token[i + 1])
                pair_counts[new_pair] = pair_counts.get(new_pair, 0) + count
            if new_pre_token:
                new_pre_token_list.append((new_pre_token, count))
        pre_token_list = new_pre_token_list
    timings["merge_seconds"] = time.perf_counter() - phase_start
    return vocab, merges, timings

def main() -> None:
    # parse arguments
    parser = argparse.ArgumentParser(description="Train a 10k BPE tokenizer on TinyStories.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "TinyStoriesV2-GPT4-train.txt",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "tokenizer" / "tinystories_10k",
    )
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument(
        "--special-token",
        action="append",
        dest="special_tokens",
        default=["<|endoftext|>"],
        help="Repeat for multiple tokens.",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found at {args.dataset}. Download TinyStories before running.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # profile training
    profile = cProfile.Profile() 
    process = psutil.Process()
    profile.enable()
    start_time = time.perf_counter()
    vocab, merges, timings = run_train_bpe_local(input_path=str(args.dataset), vocab_size=args.vocab_size, special_tokens=args.special_tokens)
    duration = time.perf_counter() - start_time; 
    profile.disable()

    # get memory usage
    rss_gb = process.memory_info().rss / (1024**3); encoder = gpt2_bytes_to_unicode()
    
    # save vocabulary and merges
    vocab_path = args.out_dir / "vocab.json"; merges_path = args.out_dir / "merges.txt"
    serialized_vocab = {}
    for idx, token in vocab.items():
        messy = []
        for b in token:
            messy.append(encoder[b])
        serialized_vocab[str(idx)] = "".join(messy)
    vocab_path.write_text(json.dumps(serialized_vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    # save merges
    merge_lines = []
    for left, right in merges:
        left_txt = "".join(encoder[b] for b in left)
        right_txt = "".join(encoder[b] for b in right)
        merge_lines.append(f"{left_txt} {right_txt}")
    merges_path.write_text("\n".join(merge_lines), encoding="utf-8")

    # get longest token
    longest_token_id = None; longest_token_len = -1; longest_text = ""
    for idx, token in vocab.items():
        if len(token) > longest_token_len:
            longest_token_len = len(token); longest_token_id = idx
            longest_text = "".join(encoder[b] for b in token)

    # get bottleneck
    stats = Stats(profile)
    if stats.stats:
        worst = max(stats.stats.items(), key=lambda item: item[1][3])
        (filename, line_no, func_name), (_, _, _, cumulative, _) = worst
        bottleneck_fn = f"{func_name} ({Path(filename).name}:{line_no})"
        bottleneck_time = cumulative
    else:
        bottleneck_fn = "unknown"; bottleneck_time = 0.0

    # save report
    report = {}
    report["dataset"] = str(args.dataset); report["vocab_size"] = len(vocab); report["num_merges"] = len(merges)
    report["elapsed_seconds"] = duration; report["elapsed_hours"] = duration / 3600; report["max_rss_gb"] = rss_gb
    report["longest_token"] = {"id": longest_token_id, "byte_length": longest_token_len, "text": longest_text}
    report["bottleneck"] = {"function": bottleneck_fn, "cumulative_seconds": bottleneck_time}
    report["phase_breakdown_seconds"] = timings

    # save report
    summary_path = args.out_dir / "training_report.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # print results
    print(f"Trained TinyStories tokenizer in {duration/60:.2f} minutes (RSS {rss_gb:.2f} GB).")
    print(f"Artifacts saved to {args.out_dir}.")
    print(f"Longest token ({longest_token_len} bytes, id {longest_token_id}): {longest_text!r}")
    print(f"Bottleneck from profiling: {bottleneck_fn} ({bottleneck_time:.2f}s cumulative).")
    for phase, seconds in timings.items():
        print(f"Phase {phase}: {seconds:.2f}s")


if __name__ == "__main__":
    main()

