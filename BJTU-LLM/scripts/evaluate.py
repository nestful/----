#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型定量/定性测试脚本

指标：
1. 关键词命中率 = 命中关键词数量 / 期望关键词数量 * 100%
2. 关键点覆盖得分 = 命中关键词数量 / 期望关键词数量 * 100%（与标签关键词一致）
3. 文本相似度 = ROUGE-L F1
4. 困惑度 = 对模型输出文本的困惑度
"""

import argparse
import json
import matplotlib.pyplot as plt
import importlib
import math
import os
import re
from difflib import SequenceMatcher

try:
    jieba = importlib.import_module("jieba")
except ImportError as exc:
    raise ImportError(
        "缺少 jieba 依赖，请先运行 `pip install jieba` 再执行测试脚本。"
    ) from exc

# 配置 HuggingFace 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
STOPWORDS = {
    # 代词和指示词
    "我们", "你们", "他们", "它们", "这个", "那个", "这些", "那些", "这里", "那里",
    # 助动词和情态动词
    "可以", "能够", "应该", "必须", "需要", "想要", "希望", "愿意", "可能", "会",
    # 连词和连接词
    "以及", "或者", "如果", "然后", "接着", "最后", "首先", "其次", "另外", "此外",
    # 常用动词（过于通用）
    "进行", "使用", "查看", "显示", "操作", "选择", "输入", "点击", "登录", "办理",
    "前往", "位于", "通过", "完成", "成功", "开始", "结束", "继续", "停止",
    # 疑问词
    "什么", "怎么", "如何", "哪里", "为什么", "哪个", "哪些", "何时", "多少",
    # 常用名词（过于通用）
    "用户", "学生", "老师", "人员", "人员", "信息", "内容", "方式", "方法", "步骤",
    "时间", "地点", "地点", "服务", "指南", "中心", "大厅", "窗口", "系统", "平台",
    # 常用修饰词
    "相关", "一般", "通常", "可能", "一定", "非常", "特别", "比较", "更加", "最好",
    # 语气词和助词
    "请", "请您", "注意", "等", "等信息", "即可", "如下", "例如", "比如",
    # 数字和量词
    "一个", "两个", "三个", "多个", "一些", "全部", "所有", "每个", "各种",
    # 其他常见无关词
    "没有", "有", "是", "不是", "在", "不在", "到", "从", "为", "被", "由",
}

CATEGORY_RULES = [
    ("Q1", 1, 15),
    ("Q2", 16, 35),
    ("Q3", 36, 45),
    ("Q4", 46, 50),
]


def configure_matplotlib():
    """配置 Matplotlib 以支持中文显示，并返回首选字体。"""
    import matplotlib.font_manager as fm
    import platform

    font_list = [f.name for f in fm.fontManager.ttflist]
    font_candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "KaiTi",
        "FangSong",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "Source Han Sans CN",
        "Source Han Sans SC",
        "PingFang SC",
        "STHeiti",
        "Hiragino Sans GB",
        "Arial Unicode MS",
    ]

    chinese_fonts = []
    font_prop = None

    for font_name in font_candidates:
        if font_name in font_list:
            chinese_fonts.append(font_name)
            if font_prop is None:
                font_prop = fm.FontProperties(family=font_name)

    if not chinese_fonts:
        system = platform.system()
        font_paths = []
        if system == "Windows":
            font_paths = [
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/simhei.ttf",
                "C:/Windows/Fonts/simsun.ttc",
            ]
        elif system == "Linux":
            font_paths = [
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/arphic/uming.ttc",
            ]
        elif system == "Darwin":
            font_paths = [
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/STHeiti Light.ttc",
            ]

        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font_prop = fm.FontProperties(fname=font_path)
                    font_name = font_prop.get_name()
                    chinese_fonts.append(font_name)
                    break
                except Exception:
                    continue

    if chinese_fonts:
        plt.rcParams["font.sans-serif"] = chinese_fonts + ["DejaVu Sans", "Arial", "sans-serif"]
        plt.rcParams["font.family"] = "sans-serif"
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS", "Arial", "sans-serif"]
        font_prop = None

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10
    return font_prop

def load_dataset(file_path: Path) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("测试文件必须是包含字典的列表")
    return data


def format_prompt(question: str) -> str:
    system_prompt = "You are a helpful assistant for BJTU campus affairs."
    return (
        "<|im_start|>system\n"
        f"{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    max_new_tokens: int = 256,
) -> str:
    prompt = format_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text


def tokenize(text: str) -> List[str]:
    tokens = [tok.strip() for tok in jieba.lcut(text) if tok.strip()]
    if tokens:
        return tokens
    # fallback
    return re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+", text)


def normalize_text(text: str) -> str:
    return " ".join(tokenize(text.lower()))


def similarity_score(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def fuzzy_contains(keyword: str, text: str, threshold: float) -> bool:
    kw = normalize_text(keyword)
    if not kw:
        return False
    normalized_text = normalize_text(text)
    if kw in normalized_text:
        return True

    kw_tokens = kw.split()
    text_tokens = normalized_text.split()
    if not text_tokens:
        return False

    min_window = max(1, len(kw_tokens) - 1)
    max_window = min(len(text_tokens), len(kw_tokens) + 2)
    candidates = set()

    for window in range(min_window, max_window + 1):
        for i in range(len(text_tokens) - window + 1):
            candidates.add(" ".join(text_tokens[i : i + window]))

    for sentence in re.split(r"[。！？!?.,;:\n]", normalized_text):
        s = sentence.strip()
        if s:
            candidates.add(s)

    for cand in candidates:
        if similarity_score(kw, cand) >= threshold:
            return True
    return False


def extract_candidate_keywords(response: str, max_candidates: int, expected_keywords: List[str] = None) -> List[str]:
    """
    从响应中提取候选关键词
    
    Args:
        response: 模型响应文本
        max_candidates: 最大候选词数量
        expected_keywords: 预期关键词列表，用于优先提取相关词
    
    Returns:
        候选关键词列表
    """
    tokens = tokenize(response)
    seen = set()
    candidates = []
    
    # 如果提供了预期关键词，优先提取与预期关键词相关的词
    if expected_keywords:
        # 先提取与预期关键词相关的词（包含或相似）
        keyword_tokens = set()
        for kw in expected_keywords:
            if kw:
                kw_tokens = tokenize(kw.lower())
                keyword_tokens.update(kw_tokens)
        
        # 第一轮：优先提取与预期关键词相关的词
        for tok in tokens:
            t = tok.strip()
            if (
                not t
                or len(t) < 2
                or t in seen
                or t in STOPWORDS
                or t.isdigit()
            ):
                continue
            
            # 检查是否与预期关键词相关
            t_lower = t.lower()
            is_related = False
            for kw_token in keyword_tokens:
                if t_lower in kw_token or kw_token in t_lower:
                    is_related = True
                    break
                # 检查相似度
                if similarity_score(t_lower, kw_token) >= 0.6:  # 使用较低的阈值
                    is_related = True
                    break
            
            if is_related:
                seen.add(t)
                candidates.append(t)
                if max_candidates and len(candidates) >= max_candidates:
                    break
        
        # 如果相关词不够，再补充其他词
        if len(candidates) < max_candidates:
            for tok in tokens:
                t = tok.strip()
                if (
                    not t
                    or len(t) < 2
                    or t in seen
                    or t in STOPWORDS
                    or t.isdigit()
                ):
                    continue
                seen.add(t)
                candidates.append(t)
                if len(candidates) >= max_candidates:
                    break
    else:
        # 原有逻辑：按顺序提取
        for tok in tokens:
            t = tok.strip()
            if (
                not t
                or len(t) < 2
                or t in seen
                or t in STOPWORDS
                or t.isdigit()
            ):
                continue
            seen.add(t)
            candidates.append(t)
            if max_candidates and len(candidates) >= max_candidates:
                break
    
    if not candidates:
        fallback = response.strip()
        if fallback:
            candidates.append(fallback)
    return candidates


def keyword_precision_recall(
    response: str, keywords: List[str], threshold: float
) -> Tuple[float, float]:
    if not keywords:
        return 100.0, 100.0

    # KPR 计算：召回率 = 匹配到的预期关键词数 / 总预期关键词数
    matched_keywords = {
        kw
        for kw in keywords
        if kw and fuzzy_contains(kw, response, threshold)
    }
    recall = len(matched_keywords) / len(keywords) * 100

    # KHP 计算：准确率 = 匹配到的候选词数 / 总候选词数
    # 优化：优先提取与预期关键词相关的候选词，减少无关词
    candidate_limit = max(len(keywords), 5)
    candidates = extract_candidate_keywords(response, candidate_limit, expected_keywords=keywords)
    
    if not candidates:
        # 如果没有提取出候选词，且确实有预期关键词，则准确率为0
        precision = 0.0
    else:
        matched_candidates = 0
        for cand in candidates:
            is_hit = False
            cand_lower = cand.lower()
            
            # 遍历所有预期关键词，检查当前候选词是否有效
            for kw in keywords:
                if not kw:
                    continue
                kw_lower = kw.lower()
                
                # 匹配策略1: 精确匹配
                if cand_lower == kw_lower:
                    is_hit = True
                    break
                
                # 匹配策略2: 子串包含（候选词包含在关键词中，或关键词包含在候选词中）
                if cand_lower in kw_lower or kw_lower in cand_lower:
                    is_hit = True
                    break
                
                # 匹配策略3: 分词后检查是否有重叠的词
                cand_tokens = set(tokenize(cand_lower))
                kw_tokens = set(tokenize(kw_lower))
                if cand_tokens & kw_tokens:  # 有交集
                    is_hit = True
                    break
                
                # 匹配策略4: 模糊相似度匹配（使用原始阈值）
                if similarity_score(cand, kw) >= threshold:
                    is_hit = True
                    break
                
                # 匹配策略5: 更宽松的相似度匹配（用于处理部分匹配）
                if similarity_score(cand, kw) >= max(0.5, threshold - 0.2):
                    # 额外检查：如果候选词长度合理（至少是关键词的一半）
                    if len(cand_lower) >= len(kw_lower) * 0.5:
                        is_hit = True
                        break
            
            if is_hit:
                matched_candidates += 1
        
        # 计算准确率
        precision = matched_candidates / len(candidates) * 100 if candidates else 0.0

    return precision, recall


def rouge_l_score(response: str, reference: str) -> float:
    try:
        rouge_module = importlib.import_module("rouge_score.rouge_scorer")
    except ImportError as exc:
        raise ImportError(
            "缺少 rouge-score 依赖，请先运行 `pip install rouge-score` 再执行测试脚本。"
        ) from exc
    RougeScorer = getattr(rouge_module, "RougeScorer")
    scorer = RougeScorer(["rougeL"], use_stemmer=True)
    ref_tokens = " ".join(tokenize(reference))
    resp_tokens = " ".join(tokenize(response))
    score = scorer.score(ref_tokens, resp_tokens)["rougeL"].fmeasure
    return score * 100


def conditional_perplexity(
    question: str,
    response: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> float:
    if not response.strip():
        return float("nan")

    prompt = format_prompt(question)
    eos = tokenizer.eos_token or ""

    prompt_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    response_inputs = tokenizer(
        response + eos,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = torch.cat(
        [prompt_inputs["input_ids"], response_inputs["input_ids"]], dim=1
    )
    attention_mask = torch.cat(
        [prompt_inputs["attention_mask"], response_inputs["attention_mask"]], dim=1
    )

    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    labels = input_ids.clone()
    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels[:, :prompt_len] = -100

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
    return math.exp(loss.item())


def evaluate_sample(
    item: Dict,
    response: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    keyword_threshold: float,
) -> Dict:
    keywords = item.get("expected_keywords", [])
    reference = item.get("reference", "")
    precision, recall = keyword_precision_recall(response, keywords, keyword_threshold)

    return {
        "id": item.get("id"),
        "question": item.get("question"),
        "response": response,
        "keyword_precision": precision,  # KHP
        "keyword_recall": recall,  # KPR
        "rouge_l": rouge_l_score(response, reference) if reference else float("nan"),
        "perplexity": conditional_perplexity(
            item.get("question", ""), response, model, tokenizer
        ),
    }


def summarize_results(records: List[Dict]) -> Dict:
    def avg(field: str) -> float:
        values = [
            r[field]
            for r in records
            if isinstance(r.get(field), (int, float)) and not math.isnan(r[field])
        ]
        return sum(values) / len(values) if values else float("nan")

    return {
        "samples": len(records),
        "avg_keyword_precision": avg("keyword_precision"),  # KHP
        "avg_keyword_recall": avg("keyword_recall"),        # KPR
        "avg_rouge_l": avg("rouge_l"),
        "avg_perplexity": avg("perplexity"),
    }


def categorize_records(records: List[Dict]) -> Dict[str, List[Dict]]:
    categories = {name: [] for name, _, _ in CATEGORY_RULES}
    categories.setdefault("其他", [])
    for rec in records:
        sid = rec.get("id")
        assigned = False
        if isinstance(sid, int):
            for name, start, end in CATEGORY_RULES:
                if start <= sid <= end:
                    categories[name].append(rec)
                    assigned = True
                    break
        if not assigned:
            categories["其他"].append(rec)
    return categories


def average_metric(records: List[Dict], field: str) -> float:
    values = [
        r[field]
        for r in records
        if field in r and isinstance(r[field], (int, float)) and not math.isnan(r[field])
    ]
    return sum(values) / len(values) if values else float("nan")


def compute_category_metrics(records: List[Dict]) -> Dict[str, Dict[str, float]]:
    categorized = categorize_records(records)
    metrics = {}
    for name, recs in categorized.items():
        if not recs:
            metrics[name] = {
                "keyword_precision": float("nan"),
                "keyword_recall": float("nan"),
                "rouge_l": float("nan"),
                "perplexity": float("nan"),
            }
        else:
            metrics[name] = {
                "keyword_precision": average_metric(recs, "keyword_precision"),
                "keyword_recall": average_metric(recs, "keyword_recall"),
                "rouge_l": average_metric(recs, "rouge_l"),
                "perplexity": average_metric(recs, "perplexity"),
            }
    return metrics


def save_results(output_dir: Path, summary: Dict, details: List[Dict], category_metrics: Dict[str, Dict[str, float]]):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / "metrics_summary.json"
    details_file = output_dir / "details.json"
    category_file = output_dir / "category_metrics.json"

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(details_file, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)

    with open(category_file, "w", encoding="utf-8") as f:
        json.dump(category_metrics, f, indent=2, ensure_ascii=False)

    print(f"✓ 评估摘要已保存至 {summary_file}")
    print(f"✓ 评估明细已保存至 {details_file}")
    print(f"✓ 分类指标已保存至 {category_file}")


def plot_metrics(category_metrics: Dict[str, Dict[str, float]], chart_root: Path, run_name: str):
    chart_root.mkdir(parents=True, exist_ok=True)
    chart_dir = chart_root / run_name
    chart_dir.mkdir(exist_ok=True)

    configure_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    axes = axes.flatten()

    metrics = [
        ("关键词命中率(KHP) (↑)", "keyword_precision"),
        ("关键词召回率(KPR) (↑)", "keyword_recall"),
        ("ROUGE-L (↑)", "rouge_l"),
        ("PPL (↓)", "perplexity"),
    ]

    category_names = [name for name, _, _ in CATEGORY_RULES]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

    for ax, (title, key) in zip(axes, metrics):
        values = [
            category_metrics.get(name, {}).get(key, float("nan"))
            for name in category_names
        ]
        ax.bar(category_names, values, color=colors[: len(category_names)])
        ax.set_title(title)
        ax.set_ylabel("得分")

        # 设置坐标轴范围：KHP/KPR/ROUGE-L 为 0-100，PPL 为 0-5
        if key == "perplexity":
            ax.set_ylim(0, 3)
        else:
            ax.set_ylim(0, 100)

        # 横轴标签保持水平展示
        ax.tick_params(axis="x", rotation=0)
        for idx, val in enumerate(values):
            if val is not None and not math.isnan(val):
                ax.text(idx, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    chart_path = chart_dir / "metrics_overview.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ 指标图已保存至 {chart_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="模型定量测试脚本")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(Path("output") / "qwen-1.5b-lora" / "best_model"),
        help="LoRA 模型目录",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="基础模型名称或路径",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=str(Path("Data") / "test" / "bjtu_test.json"),
        help="测试数据文件",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path("output") / "evaluation"),
        help="评估结果输出目录",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--keyword_threshold",
        type=float,
        default=0.8,
        help="关键词模糊匹配阈值（0-1，值越大匹配越严格）",
    )
    parser.add_argument(
        "--test_base_model",
        action="store_true",
        help="仅使用基础模型进行测试（不加载 LoRA 适配器）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("✓ 正在加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✓ 正在加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    if args.test_base_model:
        print("✓ 使用基础模型进行测试（跳过 LoRA 适配器）")
        model = base_model
    else:
        print("✓ 正在加载 LoRA 适配器...")
        model = PeftModel.from_pretrained(base_model, args.model_dir)
    model.eval()

    data = load_dataset(Path(args.data_file))
    run_name = datetime.now().strftime("eval_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_name

    records = []
    for item in data:
        question = item.get("question", "")
        if not question:
            continue
        response = generate_response(
            model, tokenizer, question, max_new_tokens=args.max_new_tokens
        )
        record = evaluate_sample(
            item, response, model, tokenizer, args.keyword_threshold
        )
        records.append(record)
        print(
            f"[ID {item.get('id')}] "
            f"KHP: {record['keyword_precision']:.2f}% | "
            f"KPR: {record['keyword_recall']:.2f}% | "
            f"ROUGE-L: {record['rouge_l']:.2f} | "
            f"PPL: {record['perplexity']:.2f}"
        )

    summary = summarize_results(records)
    category_metrics = compute_category_metrics(records)
    save_results(output_dir, summary, records, category_metrics)
    plot_metrics(category_metrics, Path("output") / "Chart", run_name)

    print("评估摘要：")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  - {k}: {v:.4f}")
        else:
            print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()