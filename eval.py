import argparse
import random
import warnings
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from model.MokioModel import MokioMindConfig, MokioMindForCausalLM
from trainer.trainer_utils import setup_seed

warnings.filterwarnings("ignore")

try:
    from model.model_lora import apply_lora, load_lora
except ImportError:
    apply_lora = None
    load_lora = None


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_weight_path(args) -> Path:
    save_dir = PROJECT_ROOT / args.save_dir
    moe_suffix = "_moe" if args.use_moe else ""

    requested_name = args.weight
    if requested_name.endswith(".pth"):
        requested = save_dir / requested_name
    else:
        requested = save_dir / f"{requested_name}_{args.hidden_size}{moe_suffix}.pth"
        if not requested.exists():
            requested = save_dir / f"{requested_name}.pth"

    if requested.exists():
        return requested

    prefix = f"{args.weight}_{args.hidden_size}{moe_suffix}"
    candidates = list(save_dir.glob(f"{prefix}*.pth"))
    if not candidates:
        candidates = list(save_dir.glob(f"*_{args.hidden_size}{moe_suffix}*.pth"))

    if candidates:
        if args.prefer_latest_checkpoint:
            chosen = max(candidates, key=lambda path: path.stat().st_mtime)
        else:
            chosen = sorted(candidates)[0]
        print(
            f"Requested weight not found: {requested}. "
            f"Falling back to: {chosen.name}"
        )
        return chosen

    raise FileNotFoundError(
        f"No checkpoint found in {save_dir} matching hidden_size={args.hidden_size}, "
        f"use_moe={args.use_moe}."
    )


def init_model(args):
    tokenizer_path = str((PROJECT_ROOT / args.load_from).resolve())
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if args.load_from == "model":
        checkpoint_path = resolve_weight_path(args)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        vocab_size = state_dict["model.embed_tokens.weight"].shape[0]
        model = MokioMindForCausalLM(
            MokioMindConfig(
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                vocab_size=vocab_size,
                use_moe=bool(args.use_moe),
                inference_rope_scaling=args.inference_rope_scaling,
            )
        )

        state_dict = {
            key: value.to(args.device) if torch.is_tensor(value) else value
            for key, value in state_dict.items()
        }
        model.load_state_dict(state_dict, strict=True)

        if args.lora_weight != "None":
            if apply_lora is None or load_lora is None:
                raise ImportError(
                    "LoRA support was requested, but model/model_lora.py is missing."
                )
            apply_lora(model)
            lora_path = (
                PROJECT_ROOT
                / args.save_dir
                / "lora"
                / f"{args.lora_weight}_{args.hidden_size}.pth"
            )
            load_lora(model, str(lora_path))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.load_from, trust_remote_code=True
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.eval().to(args.device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {total_params:.2f}M")
    return model, tokenizer


def build_parser():
    parser = argparse.ArgumentParser(description="Run local MokioMind evaluation.")
    parser.add_argument("--load_from", default="model", type=str)
    parser.add_argument("--save_dir", default="out", type=str)
    parser.add_argument(
        "--weight",
        default="pretrain",
        type=str,
        help="Checkpoint prefix or full checkpoint filename, for example: pretrain or pretrain_512_step294900.pth",
    )
    parser.add_argument(
        "--prefer_latest_checkpoint",
        default=1,
        type=int,
        choices=[0, 1],
        help="When the exact requested checkpoint is missing, prefer the newest matching checkpoint.",
    )
    parser.add_argument(
        "--lora_weight",
        default="None",
        type=str,
        help="Optional LoRA checkpoint prefix. Use 'None' to disable.",
    )
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument(
        "--inference_rope_scaling", default=False, action="store_true"
    )
    parser.add_argument("--max_new_tokens", default=512, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--repetition_penalty", default=1.1, type=float)
    parser.add_argument("--historys", default=0, type=int)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        type=str,
    )
    parser.add_argument(
        "--input_mode",
        default=-1,
        type=int,
        choices=[-1, 0, 1],
        help="-1 asks interactively, 0 runs built-in prompts, 1 uses manual input.",
    )
    return parser


def build_input_text(tokenizer, conversation, prompt, weight_name: str) -> str:
    # Pretrain checkpoints usually learn next-token continuation, while
    # chat/instruction checkpoints should use the tokenizer chat template.
    if weight_name.startswith("pretrain"):
        return f"{tokenizer.bos_token}{prompt}\nAssistant:"
    return tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )


def choose_input_mode(cli_value: int) -> int:
    if cli_value in (0, 1):
        return cli_value
    while True:
        value = input("[0] auto prompts\n[1] manual input\n").strip()
        if value in {"0", "1"}:
            return int(value)
        print("Please enter 0 or 1.")


def main():
    args = build_parser().parse_args()

    prompts = [
        "What are your strengths?",
        "Why is the sky blue?",
        "Write a Python function that returns Fibonacci numbers.",
        'Explain the concept of "photosynthesis".',
        "If it rains tomorrow, how should I prepare before going out?",
        "Compare cats and dogs as pets.",
        "Explain what machine learning is.",
        "Recommend some Chinese foods.",
    ]

    conversation = []
    model, tokenizer = init_model(args)
    input_mode = choose_input_mode(args.input_mode)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    prompt_iter = prompts if input_mode == 0 else iter(lambda: input("User: "), "")
    for prompt in prompt_iter:
        setup_seed(2026 if input_mode == 0 else random.randint(0, 2048))
        if input_mode == 0:
            print(f"User: {prompt}")

        conversation = conversation[-args.historys :] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        input_text = build_input_text(tokenizer, conversation, prompt, args.weight)
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(
            args.device
        )

        print("Assistant: ", end="")
        generated_ids = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
        )
        response = tokenizer.decode(
            generated_ids[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
        )
        conversation.append({"role": "assistant", "content": response})
        print("\n")


if __name__ == "__main__":
    main()
