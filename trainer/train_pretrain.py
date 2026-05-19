import os
import sys


__package__ = "trainer"
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

import argparse  # 命令行参数解析
import time  # 时间统计
import warnings  # 警告控制
import torch
import torch.distributed as dist  # 分布式训练支持
from contextlib import nullcontext  # 上下文管理器
from torch import optim  # 优化器
from torch.nn.parallel import DistributedDataParallel  # 分布式数据并行
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载器

from model.MokioModel import MokioMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import (  # 训练工具函数
    get_lr,
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    init_model,
    SkipBatchSampler,
)

# 忽略警告信息，保持输出清洁
warnings.filterwarnings("ignore")


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()  # 记录开始时间

    # 遍历数据批次
    for step, batch in enumerate(loader, start=start_step + 1):
        if isinstance(batch, (list, tuple)):
            input_ids = batch[0]
        else:
            input_ids = batch
        input_ids = input_ids.to(args.device, non_blocking=True)
        labels = input_ids.clone()
        labels[input_ids == pad_token_id] = -100
        # 预训练阶段 pad 只在右侧、labels 已置 -100 忽略 pad 损失，
        # 所以 attention_mask 不再下传给模型，让 SDPA 走 is_causal 快速路径。
        # labels 鐩存帴鍦?GPU 涓婃瀯閫狅紝閬垮厤棰濆鐨?CPU labels/attention_mask 鐢熸垚鍜屼紶杈?

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
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
            current_loss = loss.item() * args.accumulation_steps  # 恢复真实损失值
            current_lr = optimizer.param_groups[-1]["lr"]  # 当前学习率

            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:"
            )

            # 记录到实验跟踪系统
            if wandb:
                wandb.log(
                    {"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min}
                )

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()  # 切换到评估模式

            # 构建保存路径
            moe_suffix = (
                "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
            )
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
            os.makedirs(args.save_dir, exist_ok=True)

            # 📚 分布式模型保存知识点
            # DDP模型需要通过.module访问真正的模型
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            # 📚 半精度保存知识点
            # 将float32参数转为float16，减少存储空间
            state_dict = {k: v.half() for k, v in state_dict.items()}
            ckp_tmp = ckp + ".tmp"
            torch.save(state_dict, ckp_tmp)
            try:
                os.replace(ckp_tmp, ckp)
            except PermissionError:
                fallback_ckp = (
                    f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}"
                    f"{moe_suffix}_step{step}.pth"
                )
                os.replace(ckp_tmp, fallback_ckp)
                Logger(
                    f"Checkpoint target is busy, saved to fallback path: {fallback_ckp}"
                )

            # 保存完整训练状态
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir="../checkpoints",  # ！修正：原"checkpoints"缺少../前缀
            )

            model.train()  # 恢复训练模式


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MokioMind Pretraining")

    # ========== 基础训练参数 ==========
    parser.add_argument(
        "--save_dir", type=str, default="../out", help="模型保存目录"
    )  # ！修正：原"out"缺少../前缀
    parser.add_argument(
        "--save_weight", default="pretrain", type=str, help="保存权重的前缀名"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="训练轮数（建议1轮zero或2-6轮充分训练）"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")

    # ========== 硬件和性能参数 ==========
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="数据加载线程数",
    )
    parser.add_argument(
        "--compile",
        type=int,
        default=0,
        choices=[0, 1],
        help="是否用 torch.compile 编译模型（0=否，1=是）",
    )

    # ========== 训练策略参数 ==========
    parser.add_argument(
        "--accumulation_steps", type=int, default=64, help="梯度累积步数"
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument(
        "--gradient_checkpointing",
        type=int,
        default=1,
        choices=[0, 1],
        help="是否启用梯度检查点，0=关闭，1=开启",
    )

    # ========== 模型架构参数 ==========
    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument(
        "--max_seq_len", default=512, type=int, help="训练的最大截断长度"
    )
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）",
    )

    # ========== 数据和恢复参数 ==========
    parser.add_argument(
        "--data_path",
        type=str,
        default="../dataset/pretrain_hq.jsonl",  # ！修正：原"dataset/..."缺少../前缀
        help="预训练数据路径",
    )
    parser.add_argument(
        "--from_weight",
        default="none",
        type=str,
        help="基于哪个权重训练，为none则从头开始",
    )
    parser.add_argument(
        "--from_resume",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否自动检测&续训（0=否，1=是）",
    )

    # ========== 实验跟踪参数 ==========
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument(
        "--wandb_project", type=str, default="MokioMind-Pretrain", help="wandb项目名"
    )

    # 解析命令行参数
    args = parser.parse_args()
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.abspath(os.path.join(SCRIPT_DIR, args.data_path))
    if not os.path.isabs(args.save_dir):
        args.save_dir = os.path.abspath(os.path.join(SCRIPT_DIR, args.save_dir))
    checkpoint_dir = os.path.abspath(os.path.join(SCRIPT_DIR, "../checkpoints"))

    # ========== 1. 初始化环境和随机种子 ==========
    """
    📚 分布式训练初始化知识点：
    - local_rank: 当前进程在本机上的GPU编号
    - 随机种子: 确保不同进程有不同但可复现的随机序列
    - 这样既保证了随机性，又保证了可复现性
    """
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"  # 分布式训练时使用对应的GPU

    # 📚 随机种子设置知识点
    # 不同进程使用不同的种子，避免数据采样完全相同
    # 42是基础种子，每个进程加上自己的rank保证不同
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查点 ==========
    """
    📚 模型配置和检查点管理：
    - 创建保存目录
    - 构建模型配置对象
    - 尝试加载断点续训数据
    """
    os.makedirs(args.save_dir, exist_ok=True)  # 确保保存目录存在

    # 创建MiniMind模型配置
    lm_config = MokioMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )

    # 📚 断点续训知识点
    # 如果开启了断点续训，尝试加载之前的训练状态
    ckp_data = (
        lm_checkpoint(
            lm_config, weight=args.save_weight, save_dir=checkpoint_dir
        )  # ！修正：原"checkpoints"缺少../前缀
        if args.from_resume == 1
        else None
    )

    # ========== 3. 设置混合精度 ==========
    """
    📚 混合精度训练知识点：
    - bfloat16: Google开发，数值范围大，更稳定
    - float16: 标准半精度，节省内存但可能溢出
    - autocast: 自动选择精度，关键运算用float32
    """
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # 在 Ampere+ 上放开 TF32，matmul 吞吐显著提升，对 LM 收敛几乎无影响
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # 用新版 torch.amp API；CPU 分支退化为 nullcontext
    autocast_ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type="cuda", dtype=dtype)
    )
    # bf16 动态范围足够大，GradScaler 不再必要（且 autocast_bf16 本身也不支持 scaler）
    use_scaler = device_type == "cuda" and args.dtype == "float16"

    # ========== 4. 配置WandB实验跟踪 ==========
    """
    📚 实验跟踪系统知识点：
    - WandB: 实验管理平台，记录训练过程
    - SwanLab: 国产替代方案
    - 支持断点续训时恢复到同一个实验
    """
    wandb = None
    if args.use_wandb and is_main_process():
        # 使用SwanLab作为WandB的替代
        import swanlab as wandb

        # 📚 实验恢复知识点
        # 如果有检查点数据，获取之前的wandb_id来恢复实验
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None  # 必须恢复到指定实验

        # 构建实验名称，包含关键超参数
        wandb_run_name = f"MokioMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(
            project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume
        )

    # ========== 5. 定义模型、数据、优化器 ==========
    """
    📚 训练组件初始化：
    - 模型: 根据配置创建MiniMind模型
    - 数据集: 加载预训练数据
    - 采样器: 分布式训练的数据分配
    - 优化器: AdamW优化器
    - 缩放器: 混合精度训练的梯度缩放
    """
    # 初始化模型和分词器
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if hasattr(model, "model"):
        model.model.gradient_checkpointing = bool(args.gradient_checkpointing)
    pad_token_id = tokenizer.pad_token_id

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # fused AdamW 在新版 PyTorch 上比 foreach 还要快一档
    try:
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, fused=True)
    except TypeError:
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if ckp_data:
        # 恢复模型参数
        model.load_state_dict(ckp_data["model"])
        # 恢复优化器状态（动量、方差估计等）
        optimizer.load_state_dict(ckp_data["optimizer"])
        # 恢复梯度缩放器状态
        scaler.load_state_dict(ckp_data["scaler"])
        # 恢复训练进度
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    if dist.is_initialized():
        # 📚 RoPE位置编码特殊处理
        # freqs_cos, freqs_sin是位置编码缓存，不需要梯度同步
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # 可选 torch.compile：一次编译换稳态吞吐（首个 step 会比较慢）
    if args.compile and device_type == "cuda":
        model = torch.compile(model)

    # 统一的 DataLoader 参数，避免重复
    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=(device_type == "cuda"),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        drop_last=True,
    )
    # prefetch_factor 在 num_workers=0 时不允许传
    if args.num_workers == 0:
        loader_kwargs.pop("prefetch_factor")

    for epoch in range(start_epoch, args.epochs):
        # 📚 分布式采样器epoch设置
        # 每个epoch设置不同的随机种子，确保数据顺序随机化
        if train_sampler:
            train_sampler.set_epoch(epoch)

        # 📚 断点续训逻辑
        if epoch == start_epoch and start_step > 0:  # 第一个epoch且存在检查点
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)), args.batch_size, start_step
            )
            loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                **{k: v for k, v in loader_kwargs.items() if k != "drop_last"},
            )
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始"
            )
            train_epoch(epoch, loader, len(loader) + start_step, start_step, wandb)
        else:  # 默认从头开始
            loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                **loader_kwargs,
            )
            train_epoch(epoch, loader, len(loader), 0, wandb)
