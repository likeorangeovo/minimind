import os
import sys
import time
import argparse
import warnings
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler


__package__ = "trainer"
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from model.MokioModel import MokioMindConfig
from dataset.sft_dataset import SFTDataset
from trainer.trainer_utils import (
    Logger,
    SkipBatchSampler,
    get_lr,
    init_distributed_mode,
    is_main_process,
    lm_checkpoint,
    setup_seed,
    init_model,
)

warnings.filterwarnings("ignore")


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()

    for step, batch in enumerate(loader, start=start_step + 1):
        input_ids = batch["input_ids"].to(args.device, non_blocking=True)
        labels = batch["labels"].to(args.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(args.device, non_blocking=True)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast_ctx:
            res = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = (res.loss + res.aux_loss) / args.accumulation_steps

        if use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % args.accumulation_steps == 0:
            if use_scaler:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]["lr"]
            eta_min = spend_time / max(step, 1) * iters // 60 - spend_time // 60
            Logger(
                f"SFT Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) "
                f"loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:"
            )
            if wandb:
                wandb.log(
                    {"sft_loss": current_loss, "lr": current_lr, "epoch_Time": eta_min}
                )

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            save_model_checkpoint(step, epoch, wandb)


def save_model_checkpoint(step, epoch, wandb=None):
    model.eval()
    moe_suffix = "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
    ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
    os.makedirs(args.save_dir, exist_ok=True)

    if isinstance(model, DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    state_dict = {k: v.half() for k, v in state_dict.items()}
    ckp_tmp = ckp + ".tmp"
    torch.save(state_dict, ckp_tmp)
    os.replace(ckp_tmp, ckp)

    lm_checkpoint(
        lm_config,
        weight=args.save_weight,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=epoch,
        step=step,
        wandb=wandb,
        save_dir=checkpoint_dir,
    )
    model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MokioMind SFT")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_data.jsonl")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--save_weight", default="sft", type=str)
    parser.add_argument("--from_weight", default="pretrain", type=str)
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--compile", type=int, default=0, choices=[0, 1])
    parser.add_argument("--accumulation_steps", type=int, default=16)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--gradient_checkpointing", type=int, default=1, choices=[0, 1])
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MokioMind-SFT")
    args = parser.parse_args()

    if not os.path.isabs(args.data_path):
        args.data_path = os.path.abspath(os.path.join(SCRIPT_DIR, args.data_path))
    if not os.path.isabs(args.save_dir):
        args.save_dir = os.path.abspath(os.path.join(SCRIPT_DIR, args.save_dir))
    checkpoint_dir = os.path.abspath(os.path.join(SCRIPT_DIR, "../checkpoints"))

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"

    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    os.makedirs(args.save_dir, exist_ok=True)

    lm_config = MokioMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )

    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir=checkpoint_dir)
        if args.from_resume == 1
        else None
    )

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    autocast_ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type="cuda", dtype=dtype)
    )
    use_scaler = device_type == "cuda" and args.dtype == "float16"

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        run_name = (
            f"MokioMind-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}"
            f"-LearningRate-{args.learning_rate}"
        )
        wandb.init(
            project=args.wandb_project, name=run_name, id=wandb_id, resume=resume
        )

    model, tokenizer = init_model(
        lm_config,
        args.from_weight,
        save_dir=args.save_dir,
        device=args.device,
    )
    if hasattr(model, "model"):
        model.model.gradient_checkpointing = bool(args.gradient_checkpointing)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    try:
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, fused=True)
    except TypeError:
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    if args.compile and device_type == "cuda":
        model = torch.compile(model)

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=(device_type == "cuda"),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        drop_last=True,
    )
    if args.num_workers == 0:
        loader_kwargs.pop("prefetch_factor")

    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)), args.batch_size, start_step
            )
            loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                **{k: v for k, v in loader_kwargs.items() if k != "drop_last"},
            )
            Logger(
                f"SFT Epoch [{epoch + 1}/{args.epochs}]: skip first {start_step} steps"
            )
            train_epoch(epoch, loader, len(loader) + start_step, start_step, wandb)
        else:
            loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                **loader_kwargs,
            )
            train_epoch(epoch, loader, len(loader), 0, wandb)
