#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试基础模型与微调模型在指定数据集上的表现，并将问答日志写入文件。
用法：
    python tool/test_models.py \
        --dataset ../Data/pizza_test.json \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --finetuned_dir ../output/qwen-1.5b-lora \
        --log_dir ../logs
"""
import argparse
import json
import logging
import os

# 配置 HuggingFace 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_dataset(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array of {instruction,input,output}")
    return data


def setup_logger(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"test_results_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("=" * 80)
    logging.info("启动测试流程，日志文件：%s", log_path)
    logging.info("=" * 80)
    return log_path


def build_prompt(instruction: str, context: str) -> str:
    context = context.strip() if context else "（无补充背景）"
    user_block = (
        f"Instruction：{instruction.strip()}\n"
        f"Context：{context}\n"
        "请结合上下文，给出结构清晰、分点叙述的回答。"
    )
    prompt = (
        "<|im_start|>system\n"
        "You are a professional pizza chef and baking instructor. "
        "Respond in Chinese with detailed, well-structured explanations.\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{user_block}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt


def batched_generate(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    batch_size: int = 4,
    max_new_tokens: int = 256,
    temperature: float = 0.5,
    top_p: float = 0.9,
) -> List[str]:
    results: List[str] = []
    total = len(prompts)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_prompts = prompts[start:end]
        logging.info("  - 批量推理 [%d ~ %d)", start + 1, end)
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_lengths = inputs["attention_mask"].sum(dim=1)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for i in range(end - start):
            seq = output[i]
            cut = int(input_lengths[i].item())
            generated = seq[cut:]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            results.append(text)
    return results


def load_base_model(model_name: str, device: str, torch_dtype):
    logging.info("加载基础模型：%s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.eval()
    return model


def load_finetuned_model(model_name: str, finetuned_dir: Path, device: str, torch_dtype):
    logging.info("加载微调模型：%s (base=%s)", finetuned_dir, model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    ft_model = PeftModel.from_pretrained(base_model, str(finetuned_dir))
    ft_model.eval()
    return ft_model


def main():
    parser = argparse.ArgumentParser(description="测试基础模型与微调模型表现")
    parser.add_argument("--dataset", type=str, default="../Data/val/bjtu.json", help="测试数据集路径（JSON 文件）")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="基础模型名称或路径")
    parser.add_argument("--finetuned_dir", type=str, default="../output/qwen-1.5b-lora/best_model", help="微调模型所在目录（PEFT）")
    parser.add_argument("--log_dir", type=str, default="../logs", help="日志输出目录")
    parser.add_argument("--max_samples", type=int, default=None, help="可选：只测试前N条样本")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="生成的最大token数")
    parser.add_argument("--batch_size", type=int, default=16, help="推理批大小（同时处理的样本数量）")
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    dataset_path = (Path(args.dataset) if Path(args.dataset).is_absolute() else (script_dir / args.dataset)).resolve()
    finetuned_dir = (Path(args.finetuned_dir) if Path(args.finetuned_dir).is_absolute() else (script_dir / args.finetuned_dir)).resolve()
    log_dir = (Path(args.log_dir) if Path(args.log_dir).is_absolute() else (script_dir / args.log_dir)).resolve()

    log_path = setup_logger(log_dir)
    logging.info("项目根目录：%s", project_root)
    logging.info("数据集：%s", dataset_path)
    logging.info("微调模型目录：%s", finetuned_dir)

    dataset = load_dataset(dataset_path)
    if args.max_samples is not None:
        dataset = dataset[: args.max_samples]
    logging.info("样本数量：%d", len(dataset))

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if not torch.cuda.is_available():
        raise EnvironmentError("当前环境未检测到可用的 GPU，测试流程要求在 GPU 上执行推理。")
    device = "cuda"
    major, minor = torch.cuda.get_device_capability()
    if major >= 8:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16
    gpu_name = torch.cuda.get_device_name(0)
    logging.info("检测到 GPU：%s，计算能力：%d.%d，使用数据类型：%s", gpu_name, major, minor, torch_dtype)

    samples = [
        {
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
            "prompt": build_prompt(item.get("instruction", ""), item.get("input", "")),
            "base": None,
            "finetuned": None,
        }
        for item in dataset
    ]

    base_model = load_base_model(args.base_model, device, torch_dtype)
    logging.info("开始基础模型批量推理 ...")
    base_outputs = batched_generate(
        base_model,
        tokenizer,
        [s["prompt"] for s in samples],
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=0.3,
        top_p=0.9,
    )
    for sample, output_text in zip(samples, base_outputs):
        sample["base"] = output_text
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    finetuned_model = load_finetuned_model(args.base_model, finetuned_dir, device, torch_dtype)
    logging.info("开始微调模型批量推理 ...")
    finetuned_outputs = batched_generate(
        finetuned_model,
        tokenizer,
        [s["prompt"] for s in samples],
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=0.3,
        top_p=0.9,
    )
    for sample, output_text in zip(samples, finetuned_outputs):
        sample["finetuned"] = output_text
    del finetuned_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logging.info("=" * 80)
    logging.info("测试结果汇总：")
    logging.info("=" * 80)
    for idx, sample in enumerate(samples, start=1):
        logging.info("样例 %d/%d", idx, len(samples))
        logging.info("[您]：Instruction=%s | Context=%s", sample["instruction"], sample["input"])
        logging.info("[AI (微调前)]：%s", sample["base"])
        logging.info("[AI (微调后)]：%s", sample["finetuned"])
        logging.info("-" * 80)

    logging.info("测试完成。日志已保存至：%s", log_path)


if __name__ == "__main__":
    main()

