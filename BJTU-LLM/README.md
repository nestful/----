# BJTU Campus Assistant LLM (Qwen2.5-LoRA)

本项目基于 [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) 模型，使用 LoRA (Low-Rank Adaptation) 技术进行微调，旨在构建一个专注于 BJTU（北京交通大学）校园事务的智能问答助手。

## 📊 项目效果

通过微调，模型在特定领域的问答能力显著提升。以下是训练过程及评估指标：

### 训练与损失曲线

### ![loss_curves](README.assets/loss_curves.png)



## 学习率变化曲线（learning）

![20251119_193459_learning_rate_curve](README.assets/20251119_193459_learning_rate_curve.png)



### 评估指标对比 (Base vs LoRA)

微调前：

![metrics_微调前-1](README.assets/metrics_微调前-1.png)

微调后：![metrics_微调后-2](README.assets/metrics_微调后-2.png)



主要指标提升：

- **关键词命中率 (KHP):** 显著提升，能够更准确地捕捉校园术语。
- **困惑度 (PPL):** 下降，生成的文本更加流畅自然。
- **ROUGE-L:** 提升，回答内容与参考答案的相似度更高。

消融结果图：

![ablation_group1_rank](README.assets/ablation_group1_rank.png)



![ablation_group2_target](README.assets/ablation_group2_target.png)



## 🛠️ 环境安装

1. 克隆仓库：
```bash
git clone https://github.com/your-username/BJTU-LLM-Finetuning.git
cd BJTU-LLM-Finetuning
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 🚀 快速开始

### 1. 数据准备
请确保数据放置在 `data/` 目录下，格式为 JSON 列表：
```json
[
  {
    "instruction": "如何办理校园卡？",
    "input": "",
    "output": "您好，办理校园卡请前往..."
  }
]
```

### 2. 模型微调 (Training)
运行训练脚本。脚本会自动加载 Qwen2.5-1.5B 模型并开始 LoRA 微调。

```bash
python scripts/train.py --gpu 0
```
*参数说明：可以在 `train.py` 中修改 `Config` 类来调整 `batch_size`, `learning_rate` 等参数。*

### 3. 对话测试 (Inference)

**与微调后的模型对话：**
```bash
python scripts/chat_lora.py --model_path output/qwen-1.5b-lora/best_model
```

**与原生基座模型对话（对比用）：**
```bash
python scripts/chat_base.py
```

### 4. 模型评估 (Evaluation)

**定量评估（计算 KHP, ROUGE, PPL）：**
```bash
python scripts/evaluate.py --model_dir output/qwen-1.5b-lora/best_model --data_file data/test/bjtu_test.json
```

**定性对比（生成对比日志）：**
```bash
python scripts/compare_models.py
```

## 📂 目录结构

```text
├── assets/             # 效果图表
├── data/               # 训练与测试数据
├── output/             # (自动生成) 模型权重保存路径
├── scripts/            # 源码
│   ├── train.py        # 训练脚本
│   ├── chat_lora.py    # LoRA模型推理
│   ├── evaluate.py     # 指标评估
│   └── ...
└── logs/               # 运行日志
```

## 📝 引用与致谢

- 基础模型：[Qwen2.5](https://github.com/QwenLM/Qwen2.5)
- 微调框架：[PEFT](https://github.com/huggingface/peft)

## 📄 License

Apache 2.0