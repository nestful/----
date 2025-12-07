#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 PEFT LoRA 微调 Qwen/Qwen-1.5B 模型

这个脚本展示了如何使用 LoRA (Low-Rank Adaptation) 方法
对 Qwen/Qwen-1.5B 模型进行参数高效的微调。
"""

import os
import sys
import io

# 设置标准输出和错误输出的编码为 UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import argparse

# 解析命令行参数（必须在导入 torch 之前设置 CUDA_VISIBLE_DEVICES）
parser = argparse.ArgumentParser(description='使用 LoRA 微调 Qwen/Qwen-1.5B 模型')
parser.add_argument('--gpu', type=int, default=3, help='指定使用的 GPU ID（默认: 3）')
args, unknown = parser.parse_known_args()  # 使用 parse_known_args 避免与后续参数冲突

# 设置 CUDA_VISIBLE_DEVICES（在导入 torch 之前）
if 'CUDA_VISIBLE_DEVICES' not in os.environ:  # 如果环境变量未设置，则使用命令行参数
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print(f"已设置 CUDA_VISIBLE_DEVICES={args.gpu}，程序将使用 GPU {args.gpu}（在 PyTorch 中显示为 cuda:0）")
else:
    print(f"检测到已设置的 CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}，将使用该设置")

# 配置 HuggingFace 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_CACHE'] = os.environ.get('HUGGINGFACE_HUB_CACHE', os.path.expanduser('~/.cache/huggingface'))
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
# 延迟导入 Trainer 相关模块，避免 datasets 导入问题
try:
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    HAS_TRAINER = True
except ImportError:
    HAS_TRAINER = False
    print("警告: 无法导入 Trainer，将使用手动训练循环")

from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import psutil


# 设置随机种子以确保可复现性
set_seed(42)

# ==================== 配置参数 ====================
# 获取脚本所在目录，用于解析相对路径
_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent  # tool/ 的父目录


def configure_matplotlib():
    """配置 Matplotlib 以支持中文显示。
    
    Returns:
        FontProperties: 找到的中文字体属性，如果没找到则返回 None
    """
    import matplotlib.font_manager as fm
    import matplotlib
    import os
    import platform
    
    # 获取所有可用字体
    font_list = [f.name for f in fm.fontManager.ttflist]
    
    # 按优先级查找中文字体（包含更多可能的字体名称）
    font_candidates = [
        # Windows 常见字体
        'Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong',
        # Linux 常见字体
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 
        'Noto Sans CJK TC', 'Source Han Sans CN', 'Source Han Sans SC',
        'WenQuanYi Micro Hei Light', 'AR PL UMing CN', 'AR PL UKai CN',
        # macOS 常见字体
        'STHeiti', 'STSong', 'PingFang SC', 'Hiragino Sans GB',
        # 其他
        'Arial Unicode MS'
    ]
    
    # 查找可用的中文字体
    chinese_fonts = []
    font_prop = None
    
    for font_name in font_candidates:
        if font_name in font_list:
            chinese_fonts.append(font_name)
            # 创建 FontProperties 对象用于后续使用
            if font_prop is None:
                font_prop = fm.FontProperties(family=font_name)
    
    # 如果没找到，尝试通过字体路径查找
    if not chinese_fonts:
        system = platform.system()
        font_paths = []
        
        if system == 'Windows':
            font_paths = [
                'C:/Windows/Fonts/msyh.ttc',  # Microsoft YaHei
                'C:/Windows/Fonts/simhei.ttf',  # SimHei
                'C:/Windows/Fonts/simsun.ttc',  # SimSun
            ]
        elif system == 'Linux':
            font_paths = [
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
                '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                '/usr/share/fonts/truetype/arphic/uming.ttc',  # AR PL UMing CN
            ]
        elif system == 'Darwin':  # macOS
            font_paths = [
                '/System/Library/Fonts/PingFang.ttc',
                '/System/Library/Fonts/STHeiti Light.ttc',
            ]
        
        # 检查字体文件是否存在
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    # 尝试添加字体
                    font_prop = fm.FontProperties(fname=font_path)
                    font_name = font_prop.get_name()
                    chinese_fonts.append(font_name)
                    print(f"✓ 通过路径找到中文字体: {font_name} ({font_path})")
                    break
                except Exception as e:
                    continue
    
    # 配置字体
    if chinese_fonts:
        # 设置字体优先级列表
        plt.rcParams['font.sans-serif'] = chinese_fonts + ['DejaVu Sans', 'Arial', 'sans-serif']
        # 确保使用第一个找到的中文字体
        plt.rcParams['font.family'] = 'sans-serif'
        print(f"✓ 已配置中文字体: {chinese_fonts[0]}")
    else:
        # 回退方案：使用 Unicode 字体或警告
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'Arial', 'sans-serif']
        print("⚠ 警告: 未找到中文字体，图表中的中文可能显示为方块")
        print("   建议安装中文字体，如：Microsoft YaHei (Windows) 或 WenQuanYi Micro Hei (Linux)")
        font_prop = None
    
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置字体大小为 12pt
    plt.rcParams['font.size'] = 10
    
    return font_prop


def plot_training_curves(log_history: List[dict], chart_dir: Path, run_name: str, logger: logging.Logger):
    """根据训练日志绘制损失/学习率曲线并保存。"""
    if not log_history:
        logger.warning("未找到训练日志，跳过曲线绘制。")
        return

    chart_dir.mkdir(parents=True, exist_ok=True)
    # 配置中文字体并获取字体属性
    chinese_font = configure_matplotlib()

    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []
    lr_steps, lrs = [], []
    fallback_step = 0

    for record in log_history:
        step = record.get("step", record.get("global_step"))
        if step is None:
            fallback_step += 1
            step = fallback_step

        if "loss" in record:
            train_steps.append(step)
            train_losses.append(record["loss"])
        if "eval_loss" in record:
            eval_steps.append(step)
            eval_losses.append(record["eval_loss"])
        if "learning_rate" in record:
            lr_steps.append(step)
            lrs.append(record["learning_rate"])

    if train_steps:
        # 设置图像宽度为 3.5 英寸，高度按比例计算（通常使用黄金比例）
        fig_width = 3.5  # 英寸
        fig_height = fig_width * 0.618  # 黄金比例，约 2.16 英寸
        plt.figure(figsize=(fig_width, fig_height))
        plt.plot(train_steps, train_losses, label="训练损失")
        if eval_steps:
            plt.plot(eval_steps, eval_losses, label="验证损失")
        plt.title("训练 / 验证损失曲线", fontproperties=chinese_font, fontsize=12)
        plt.xlabel("训练步数", fontproperties=chinese_font, fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend(prop=chinese_font, fontsize=12)
        plt.tick_params(labelsize=12)  # 设置坐标轴刻度字体大小
        plt.grid(True, linestyle="--", alpha=0.5)
        loss_path = chart_dir / f"{run_name}_loss_curve.png"
        plt.savefig(loss_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"训练/验证损失曲线已保存：{loss_path}")
    else:
        logger.warning("日志中未找到训练损失信息，无法绘制损失曲线。")

    if lr_steps:
        # 设置图像宽度为 3.5 英寸，高度按比例计算（通常使用黄金比例）
        fig_width = 3.5  # 英寸
        fig_height = fig_width * 0.618  # 黄金比例，约 2.16 英寸
        plt.figure(figsize=(fig_width, fig_height))
        plt.plot(lr_steps, lrs, color="orange", label="学习率")
        plt.title("学习率变化曲线", fontproperties=chinese_font, fontsize=12)
        plt.xlabel("训练步数", fontproperties=chinese_font, fontsize=12)
        plt.ylabel("Learning Rate", fontsize=12)
        plt.legend(prop=chinese_font, fontsize=12)
        plt.tick_params(labelsize=12)  # 设置坐标轴刻度字体大小
        plt.grid(True, linestyle="--", alpha=0.5)
        lr_path = chart_dir / f"{run_name}_learning_rate_curve.png"
        plt.savefig(lr_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"学习率曲线已保存：{lr_path}")
    else:
        logger.warning("日志中未找到学习率信息，未生成学习率曲线。")


class Config:
    # 模型配置
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir = str(_PROJECT_ROOT / "output" / "qwen-1.5b-lora")  # 输出目录
    
    # LoRA 配置
    lora_r = 8  # LoRA 的秩，控制适配器的大小
    lora_alpha = 16  # LoRA 的缩放参数
    lora_dropout = 0.05  # LoRA 的 dropout 率
    lora_target_modules = "all"  # 目标模块：对所有支持的线性层应用 LoRA
    
    # 训练配置
    # 针对 600 条数据的优化配置
    num_epochs = 15  # 数据量适中，15轮足够，避免过拟合
    batch_size = 8  # 已设置
    gradient_accumulation_steps = 2  # 降低到2，有效batch_size=128，每个epoch约5步
    learning_rate = 1.5e-4  # 稍微降低学习率，更稳定
    warmup_steps = 10
    max_length = 512
    logging_steps = 5 
    
    # 数据配置（训练集和验证集分别存储）
    train_data_file = str(_PROJECT_ROOT / "Data" / "train" / "bjtu.json")  # 训练集文件路径
    val_data_file = str(_PROJECT_ROOT / "Data" / "val" / "bjtu.json")  # 验证集文件路径
    
    # 日志配置
    log_dir = str(_PROJECT_ROOT / "logs")  # 日志目录
    log_file = None  # 日志文件名（自动生成）
    
    # 设备配置
    gpu_id = 3  # 默认使用 GPU:2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = True if torch.cuda.is_available() else False  # 混合精度训练


# ==================== 数据集处理 ====================
class TextDataset(Dataset):
    """简单的文本数据集类"""
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # 对文本进行编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }


def format_instruction_data(instruction, input_text, output):
    """
    将 instruction-input-output 格式转换为 Qwen2.5-Instruct 的对话格式
    """
    # Qwen2.5-Instruct 的对话格式
    system_msg = "You are a helpful assistant."
    
    if input_text and input_text.strip():
        # 如果有 input，将其与 instruction 组合
        user_msg = f"{instruction}\n{input_text}"
    else:
        # 如果 input 为空，只使用 instruction
        user_msg = instruction
    
    # 构建完整的对话格式
    formatted_text = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
    
    return formatted_text


def setup_logging(config, run_name: Optional[str] = None):
    """
    设置日志记录
    """
    # 创建日志目录
    log_dir = Path(config.log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # 生成日志文件名（带时间戳）
    timestamp = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    config.log_file = str(log_file)
    
    # 配置日志格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 配置日志：同时输出到文件和控制台
    # 创建文件处理器（确保使用 UTF-8 编码）
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # 创建控制台处理器（确保使用 UTF-8 编码）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # 配置根日志记录器
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件: {log_file}")
    return logger, timestamp


def log_config(logger, config):
    """
    记录配置参数
    """
    logger.info("=" * 60)
    logger.info("训练配置参数")
    logger.info("=" * 60)
    logger.info(f"模型名称: {config.model_name}")
    logger.info(f"输出目录: {config.output_dir}")
    logger.info(f"训练集文件: {config.train_data_file}")
    logger.info(f"验证集文件: {config.val_data_file}")
    logger.info("")
    logger.info("LoRA 配置:")
    logger.info(f"  - r (rank): {config.lora_r}")
    logger.info(f"  - alpha: {config.lora_alpha}")
    logger.info(f"  - dropout: {config.lora_dropout}")
    logger.info(f"  - target_modules: {config.lora_target_modules}")
    logger.info("")
    logger.info("训练配置:")
    logger.info(f"  - 训练轮数: {config.num_epochs}")
    logger.info(f"  - 批次大小: {config.batch_size}")
    logger.info(f"  - 梯度累积步数: {config.gradient_accumulation_steps}")
    logger.info(f"  - 学习率: {config.learning_rate}")
    logger.info(f"  - Warmup 步数: {config.warmup_steps}")
    logger.info(f"  - 最大序列长度: {config.max_length}")
    logger.info(f"  - 日志步数: {config.logging_steps}")
    logger.info(f"  - 评估策略: epoch（每个 epoch 结束后评估）")
    logger.info(f"  - 保存策略: epoch（每个 epoch 结束后保存）")
    logger.info(f"  - 加载最佳模型: 是（训练结束后自动加载最佳模型）")
    logger.info("")
    logger.info("设备配置:")
    logger.info(f"  - 设备: {config.device}")
    if config.device == "cuda":
        logger.info(f"  - GPU ID: {config.gpu_id}")
    logger.info(f"  - FP16: {config.fp16}")
    logger.info("=" * 60)


def load_json_data(file_path, logger=None):
    """
    从 JSON 文件加载数据
    """
    log_func = logger.info if logger else print
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        log_func(f"成功加载 {len(data)} 条数据从 {file_path}")
        return data
    except FileNotFoundError:
        error_msg = f"错误: 找不到数据文件 {file_path}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        raise FileNotFoundError(error_msg)
    except json.JSONDecodeError as e:
        error_msg = f"错误: JSON 解析失败: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        raise
    except Exception as e:
        error_msg = f"错误: 加载数据文件时出错: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        raise


def prepare_dataset(tokenizer, data_file, max_length, logger, dataset_name="数据集"):
    """
    准备数据集（从 JSON 数据文件加载）
    
    Args:
        tokenizer: tokenizer 对象
        data_file: 数据文件路径
        max_length: 最大序列长度
        logger: 日志记录器
        dataset_name: 数据集名称（用于日志显示）
    
    Returns:
        TextDataset 对象
    """
    if not data_file:
        error_msg = f"错误: 必须指定{dataset_name}文件路径"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # 加载 JSON 数据
    data = load_json_data(data_file, logger)
    if not data:
        error_msg = f"错误: 无法加载{dataset_name}文件"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"正在处理 {dataset_name} JSON 数据...")
    texts = []
    skipped = 0
    
    for item in data:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")
        
        if instruction and output:
            formatted_text = format_instruction_data(instruction, input_text, output)
            texts.append(formatted_text)
        else:
            skipped += 1
    
    if texts:
        logger.info(f"成功处理 {len(texts)} 条{dataset_name}样本")
        if skipped > 0:
            logger.warning(f"跳过 {skipped} 条无效数据（缺少 instruction 或 output）")
    else:
        error_msg = f"错误: {dataset_name}文件中没有有效数据"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # 创建数据集
    dataset = TextDataset(texts, tokenizer, max_length)
    # 保存 texts 到 dataset 对象，以便后续推理测试使用
    dataset.texts = texts
    logger.info(f"{dataset_name}创建完成，共 {len(dataset)} 个样本")
    
    return dataset


# ==================== 主函数 ====================
def main():
    # args 已经在文件顶部解析了
    config = Config()
    config.gpu_id = args.gpu  # 保存原始 GPU ID 用于日志显示
    
    # 验证 GPU 是否可用
    if torch.cuda.is_available():
        if torch.cuda.device_count() == 0:
            print(f"警告: 设置 CUDA_VISIBLE_DEVICES={args.gpu} 后，没有可用的 GPU")
            print("请检查 GPU ID 是否正确")
        else:
            print(f"PyTorch 可见的 GPU 数量: {torch.cuda.device_count()}")
            print(f"实际使用的物理 GPU: {args.gpu}（在 PyTorch 中映射为 cuda:0）")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.run_name = run_timestamp

    # 设置日志
    logger, _ = setup_logging(config, run_timestamp)
    logger.info("=" * 60)
    logger.info("使用 LoRA 微调 Qwen/Qwen-1.5B 模型")
    logger.info("=" * 60)

    # 设置输出目录结构：与 chart 一致，使用 run_ 时间戳目录
    base_output_dir = Path(config.output_dir)
    run_dir = base_output_dir / f"run_{run_timestamp}"
    checkpoint_dir = run_dir / "checkpoint"
    best_model_dir = run_dir  # 最终模型直接保存在 run_ 时间戳目录下
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)
    chart_dir = run_dir / "charts"
    
    # 记录运行环境信息
    import os
    import psutil
    logger.info("运行环境信息:")
    logger.info(f"  - 进程 ID (PID): {os.getpid()}")
    logger.info(f"  - 主进程名称: {psutil.Process().name()}")
    logger.info(f"  - 工作目录: {os.getcwd()}")
    if torch.cuda.is_available():
        logger.info(f"  - PyTorch 可见的 GPU 数量: {torch.cuda.device_count()}")
        logger.info(f"  - 实际使用的物理 GPU ID: {config.gpu_id}（通过 CUDA_VISIBLE_DEVICES 设置）")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"  - PyTorch GPU {i} (物理 GPU {config.gpu_id}): {gpu_name} ({gpu_memory:.2f} GB) [当前使用]")
    logger.info("")
    
    # 记录配置
    log_config(logger, config)
    logger.info(f"运行目录: {run_dir}")
    logger.info(f"Checkpoint 目录: {checkpoint_dir}")
    logger.info(f"最终模型保存目录: {best_model_dir}")
    
    # 1. 加载 tokenizer
    logger.info("正在加载 tokenizer...")
    # 优先尝试使用本地缓存
    try:
        logger.info("尝试使用本地缓存...")
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            local_files_only=True
        )
        logger.info("✓ 使用本地缓存的 tokenizer")
    except Exception:
        logger.info("本地缓存不完整，从网络下载...")
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            local_files_only=False
        )
        logger.info("✓ Tokenizer 下载完成并已缓存")
    
    # 设置 pad_token（如果不存在）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info(f"✓ Tokenizer 加载成功，词汇表大小: {len(tokenizer)}")
    
    # 2. 加载模型
    logger.info("正在加载模型...")
    
    # 检查缓存状态
    cache_path = os.path.expanduser("~/.cache/huggingface/hub")
    model_cache_dir = os.path.join(
        cache_path,
        f"models--{config.model_name.replace('/', '--')}"
    )
    
    if os.path.exists(model_cache_dir):
        cache_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(model_cache_dir)
            for filename in filenames
        ) / (1024**3)  # 转换为 GB
        logger.info(f"检测到模型缓存: {cache_size:.2f} GB")
        logger.info("尝试使用本地缓存加载模型...")
    else:
        logger.info("未检测到模型缓存，将从网络下载（约 3GB）")
        logger.info(f"缓存位置: {cache_path}")
        logger.info("下载完成后会自动缓存，下次运行将直接使用缓存")
    
    # 设置 device_map
    # 由于已经设置了 CUDA_VISIBLE_DEVICES，PyTorch 只能看到指定的 GPU
    # 所以这里使用 "cuda:0" 或 "auto" 都可以，因为只有一个 GPU 可见
    if config.device == "cuda":
        device_map = "cuda:0"  # 使用 cuda:0，因为 CUDA_VISIBLE_DEVICES 已经限制了可见的 GPU
    else:
        device_map = None
    
    # 优先尝试使用本地缓存
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.fp16 else torch.float32,
            device_map=device_map,
            trust_remote_code=True,
            local_files_only=True,  # 优先使用本地缓存
        )
        logger.info("✓ 使用本地缓存的模型（无需下载）")
    except Exception as e:
        logger.info(f"本地缓存不完整或缺失: {e}")
        logger.info("从网络下载模型文件（支持断点续传）...")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.fp16 else torch.float32,
            device_map=device_map,
            trust_remote_code=True,
            local_files_only=False,  # 允许从网络下载
        )
        logger.info("✓ 模型下载完成并已缓存到本地")
    
    # 记录 GPU 使用情况
    if torch.cuda.is_available():
        current_device = next(model.parameters()).device
        logger.info(f"模型所在设备: {current_device}")
        if hasattr(model, 'hf_device_map'):
            logger.info(f"模型设备映射: {model.hf_device_map}")
        else:
            logger.info(f"使用单 GPU: {current_device}")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✓ 模型加载成功")
    logger.info(f"  - 总参数量: {total_params:,}")
    logger.info(f"  - 可训练参数量（应用 LoRA 前）: {trainable_params_before:,}")
    
    # 3. 配置 LoRA
    logger.info("正在配置 LoRA...")
    target_modules = config.lora_target_modules
    if isinstance(target_modules, str) and target_modules.lower() == "all":
        # 将“all”映射为 transformers/peft 支持的 all-linear 策略
        target_modules = "all-linear"

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    
    # 4. 应用 LoRA 到模型
    model = get_peft_model(model, peft_config)
    
    # 记录可训练参数
    logger.info("=" * 60)
    logger.info("模型参数统计:")
    logger.info("=" * 60)
    
    # 获取可训练参数信息
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_percentage = 100 * trainable_params / all_params
    
    logger.info(f"可训练参数: {trainable_params:,}")
    logger.info(f"总参数: {all_params:,}")
    logger.info(f"可训练比例: {trainable_percentage:.4f}%")
    logger.info("=" * 60)
    
    # 5. 准备数据集（分别加载训练集和验证集）
    logger.info("正在准备数据集...")
    train_dataset = prepare_dataset(
        tokenizer, 
        config.train_data_file, 
        config.max_length, 
        logger, 
        dataset_name="训练集"
    )
    eval_dataset = prepare_dataset(
        tokenizer, 
        config.val_data_file, 
        config.max_length, 
        logger, 
        dataset_name="验证集"
    )
    
    logger.info("=" * 60)
    logger.info("数据集加载完成:")
    logger.info(f"  - 训练集: {len(train_dataset)} 个样本")
    logger.info(f"  - 验证集: {len(eval_dataset)} 个样本")
    logger.info("=" * 60)
    
    # 6. 配置训练参数
    if not HAS_TRAINER:
        error_msg = "错误: 需要 Trainer 模块进行训练，请确保 transformers 库正确安装"
        logger.error(error_msg)
        return
    
    logger.info("配置训练参数...")
    
    # 按照官方样例配置 TrainingArguments
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),  # checkpoint 保存在 checkpoint 目录
        remove_unused_columns=False,  # 官方样例推荐设置
        eval_strategy="epoch",  # 每个 epoch 结束后评估（官方样例风格）
        save_strategy="epoch",  # 每个 epoch 结束后保存（官方样例风格）
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        per_device_eval_batch_size=config.batch_size,  # 官方样例包含 eval batch size
        fp16=config.fp16,  # 官方样例使用 fp16=True
        num_train_epochs=config.num_epochs,
        logging_steps=config.logging_steps,
        load_best_model_at_end=True,  # 官方样例推荐：训练结束后加载最佳模型
        label_names=["labels"],  # 官方样例包含此参数
        save_total_limit=5,  # 保留最多5个检查点
        report_to="none",  # 不使用 wandb 或 tensorboard
        dataloader_num_workers=0,  # 不使用多进程数据加载，避免创建额外进程
        warmup_steps=config.warmup_steps,
        lr_scheduler_type="cosine",  # 使用余弦学习率调度，更平滑
    )
    
    # 记录训练配置中的进程信息
    logger.info("训练进程配置:")
    logger.info(f"  - DataLoader 工作进程数: {training_args.dataloader_num_workers}")
    logger.info(f"  - 分布式训练: 否（单进程训练）")
    logger.info("")
    
    # 7. 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 因果语言模型不使用 MLM
    )
    
    # 8. 创建 Trainer（按照官方样例，包含验证集）
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # 官方样例包含验证集
        data_collator=data_collator,
    )
    
    # 9. 开始训练
    logger.info("=" * 60)
    logger.info("开始训练...")
    logger.info("=" * 60)
    
    import time
    start_time = time.time()
    
    try:
        train_result = trainer.train()
        training_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("训练完成！")
        logger.info("=" * 60)
        logger.info(f"训练耗时: {training_time/3600:.2f} 小时 ({training_time:.2f} 秒)")
        logger.info(f"最终训练损失: {train_result.training_loss:.4f}")
        
        # 记录训练指标
        if hasattr(train_result, 'metrics'):
            logger.info("训练指标:")
            for key, value in train_result.metrics.items():
                logger.info(f"  - {key}: {value}")
        
    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    # 10. 保存最终模型到 best_model 目录
    logger.info("=" * 60)
    logger.info("保存最终模型...")
    logger.info("=" * 60)
    
    best_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    
    logger.info(f"最终模型已保存到: {best_model_dir}")
    
    # 保存训练配置到 JSON
    config_dict = {
        "model_name": config.model_name,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "lora_target_modules": config.lora_target_modules,
        "num_epochs": config.num_epochs,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "warmup_steps": config.warmup_steps,
        "max_length": config.max_length,
        "train_data_file": config.train_data_file,
        "val_data_file": config.val_data_file,
        "checkpoint_dir": str(checkpoint_dir),
        "best_model_dir": str(best_model_dir),
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percentage": trainable_percentage,
        "training_time_seconds": training_time,
        "final_loss": train_result.training_loss if 'train_result' in locals() else None,
        "run_name": run_timestamp,
    }
    
    config_file = best_model_dir / "training_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"训练配置已保存到: {config_file}")

    plot_training_curves(trainer.state.log_history, chart_dir, run_timestamp, logger)
    
    # 11. 测试推理
    logger.info("=" * 60)
    logger.info("测试推理...")
    logger.info("=" * 60)
    
    try:
        # 加载保存的模型进行推理
        logger.info("加载微调后的模型进行推理测试...")
        # 设置 device_map（与训练时保持一致）
        # 由于已经设置了 CUDA_VISIBLE_DEVICES，使用 cuda:0 即可
        if config.device == "cuda":
            inference_device_map = "cuda:0"
        else:
            inference_device_map = None
        
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.fp16 else torch.float32,
            device_map=inference_device_map,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, best_model_dir)
        model.eval()
        logger.info("✓ 模型加载成功")
        
        # 从训练集中选择几个测试样本
        test_samples = []
        # 获取训练集的原始 texts（通过 dataset 对象）
        train_texts = train_dataset.texts if hasattr(train_dataset, 'texts') else []
        if len(train_texts) > 0:
            # 选择前3个样本进行测试
            for i in range(min(3, len(train_texts))):
                sample_text = train_texts[i]
                # 提取 instruction（简化处理）
                if "<|im_start|>user" in sample_text and "<|im_end|>" in sample_text:
                    user_part = sample_text.split("<|im_start|>user")[1].split("<|im_end|>")[0].strip()
                    test_samples.append(user_part)
        
        # 如果没有提取到，使用默认测试
        if not test_samples:
            test_samples = ["什么门永远关不上？", "什么人一年只上一天班？", "什么蛋中看不中吃？"]
        
        logger.info(f"测试 {len(test_samples)} 个样本:")
        logger.info("")
        
        inference_results = []
        for i, test_prompt in enumerate(test_samples, 1):
            logger.info(f"测试 {i}/{len(test_samples)}: {test_prompt}")
            
            # 构建完整的对话格式
            full_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{test_prompt}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            # 提取 assistant 的回复
            if "<|im_start|>assistant" in generated_text:
                assistant_response = generated_text.split("<|im_start|>assistant")[1].split("<|im_end|>")[0].strip()
            else:
                assistant_response = generated_text[len(full_prompt):].strip()
            
            logger.info(f"  输入: {test_prompt}")
            logger.info(f"  输出: {assistant_response}")
            logger.info("")
            
            inference_results.append({
                "prompt": test_prompt,
                "response": assistant_response
            })
        
        # 保存推理结果
        results_file = best_model_dir / "inference_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(inference_results, f, indent=2, ensure_ascii=False)
        logger.info(f"推理结果已保存到: {results_file}")
        
    except Exception as e:
        logger.error(f"推理测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("=" * 60)
    logger.info("训练和测试完成！")
    logger.info(f"日志文件: {config.log_file}")
    logger.info(f"运行目录: {run_dir}")
    logger.info(f"Checkpoint 目录: {checkpoint_dir}")
    logger.info(f"最终模型目录: {best_model_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

