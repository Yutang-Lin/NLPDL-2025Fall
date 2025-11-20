#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Optional

import torch

from basics.templates import TransformerLM, LSTMLM
from basics.tokenizer import Tokenizer
from basics.templates import load_checkpoint


def build_model_from_config(config: dict, device: str) -> tuple[torch.nn.Module, str, torch.dtype]:
    model_type = config.get("model_type", "transformer").lower()
    dtype_str = config.get("dtype", "float32")
    dtype = getattr(torch, dtype_str)
    model_config = dict(config.get("model", {}))
    model_config["vocab_size"] = config.get("vocab_size", model_config.get("vocab_size", 50257))
    model_config["device"] = device
    model_config["dtype"] = dtype
    if model_type == "transformer":
        model_config.setdefault("context_length", config.get("context_length", 1024))
        model_config.setdefault("num_layers", 12)
        model_config.setdefault("d_model", 768)
        model_config.setdefault("num_heads", 12)
        model_config.setdefault("d_ff", 3072)
        model = TransformerLM(**model_config)
    else:
        model_config.setdefault("d_model", 768)
        model_config.setdefault("num_layers", 12)
        model = LSTMLM(**model_config)
    return model, model_type, dtype


def get_eos_token_id(tokenizer: Tokenizer, eos_token: Optional[str]) -> Optional[int]:
    if eos_token is None:
        return None
    eos_ids = tokenizer.encode(eos_token)
    if not eos_ids:
        return None
    return eos_ids[0]


def decode_text(
    model: torch.nn.Module,
    model_type: str,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: Optional[int],
    device: str,
) -> str:
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        raise ValueError("Prompt produced no tokens; please provide a non-empty prompt.")
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated = model.decode(
        input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
    )
    return tokenizer.decode(generated[0].tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode text from a trained language model.")
    parser.add_argument("--config", required=True, help="Path to JSON config used for training.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--vocab", required=True, help="Path to tokenizer vocab JSON.")
    parser.add_argument("--merges", required=True, help="Path to tokenizer merges file.")
    parser.add_argument(
        "--prompt",
        default="Once upon a time",
        help="Prompt text to start generation.",
    )
    parser.add_argument("--prompt-file", default=None, help="Optional path to a file containing the prompt.")
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"], help="Special tokens.")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p threshold.")
    parser.add_argument("--eos-token", default="<|endoftext|>", help="End-of-sequence token.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text())
    model, model_type, _ = build_model_from_config(config, args.device)
    load_checkpoint(args.checkpoint, model)
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab,
        merges_filepath=args.merges,
        special_tokens=args.special_token,
    )
    if args.prompt_file is not None:
        prompt_texts = Path(args.prompt_file).read_text().split("\n")
    else:
        prompt_texts = [args.prompt]
    eos_token_id = get_eos_token_id(tokenizer, args.eos_token)
    model.to(args.device)
    for prompt_text in prompt_texts:
        generated_text = decode_text(
            model=model,
            model_type=model_type,
            tokenizer=tokenizer,
            prompt=prompt_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=eos_token_id,
            device=args.device,
        )
        print("=" * 80)
        print("Prompt:")
        print(prompt_text.strip())
        print("-" * 80)
        print("Generated:")
        print(generated_text.strip())
        print("=" * 80)


if __name__ == "__main__":
    main()

