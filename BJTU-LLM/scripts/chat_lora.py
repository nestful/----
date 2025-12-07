#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
交互式对话脚本 - 使用训练好的 LoRA 模型

使用方法:
    python chat_with_model.py
    python chat_with_model.py --model_path output/qwen-1.5b-lora
"""

import os
import sys
import io
import argparse

# 设置标准输出和错误输出的编码为 UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 解析命令行参数（必须在导入 torch 之前设置 CUDA_VISIBLE_DEVICES）
parser = argparse.ArgumentParser(description='使用训练好的 LoRA 模型进行交互式对话')
parser.add_argument('--gpu', type=int, default=3, help='指定使用的 GPU ID（默认: 3）')
args_pre, unknown = parser.parse_known_args()  # 使用 parse_known_args 避免与后续参数冲突

# 设置 CUDA_VISIBLE_DEVICES（在导入 torch 之前）
if 'CUDA_VISIBLE_DEVICES' not in os.environ:  # 如果环境变量未设置，则使用命令行参数
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args_pre.gpu)
    print(f"已设置 CUDA_VISIBLE_DEVICES={args_pre.gpu}，程序将使用 GPU {args_pre.gpu}（在 PyTorch 中显示为 cuda:0）")
else:
    print(f"检测到已设置的 CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}，将使用该设置")

# 配置 HuggingFace 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_CACHE'] = os.environ.get('HUGGINGFACE_HUB_CACHE', os.path.expanduser('~/.cache/huggingface'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
import logging
from datetime import datetime
import time


class ChatBot:
    """交互式对话机器人"""
    
    def __init__(self, model_path, base_model_name=None, device=None, max_length=2048, log_dir="logs", gpu_id=None):
        """
        初始化对话机器人
        
        Args:
            model_path: LoRA 模型路径
            base_model_name: 基础模型名称（如果为 None，从 adapter_config.json 读取）
            device: 设备（cuda/cpu），如果为 None 自动选择
            max_length: 最大生成长度
            log_dir: 日志目录
            gpu_id: GPU ID（如果为 None，使用默认值或环境变量设置）
        """
        self.model_path = Path(model_path)
        self.max_length = max_length
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.log_dir = Path(log_dir)
        self.gpu_id = gpu_id  # 保存 GPU ID 用于日志显示
        
        # 设置日志
        self._setup_logging()
        
        print("=" * 60)
        print("正在加载模型...")
        print("=" * 60)
        
        # 1. 读取基础模型名称
        if base_model_name is None:
            adapter_config_path = self.model_path / "adapter_config.json"
            if adapter_config_path.exists():
                import json
                with open(adapter_config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    base_model_name = config.get("base_model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")
                print(f"从配置文件读取基础模型: {base_model_name}")
            else:
                base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
                print(f"使用默认基础模型: {base_model_name}")
        else:
            print(f"使用指定的基础模型: {base_model_name}")
        
        # 2. 加载 tokenizer
        print("\n正在加载 tokenizer...")
        tokenizer_path = self.model_path if (self.model_path / "tokenizer.json").exists() else base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            local_files_only=False
        )
        
        # 设置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print(f"✓ Tokenizer 加载成功，词汇表大小: {len(self.tokenizer)}")
        
        # 3. 加载基础模型
        print("\n正在加载基础模型...")
        # 设置 device_map（由于已经设置了 CUDA_VISIBLE_DEVICES，使用 cuda:0 即可）
        if self.device == "cuda":
            device_map = "cuda:0"  # 使用 cuda:0，因为 CUDA_VISIBLE_DEVICES 已经限制了可见的 GPU
        else:
            device_map = None
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=device_map,
            trust_remote_code=True,
            local_files_only=False
        )
        print("✓ 基础模型加载成功")
        
        # 4. 加载 LoRA 适配器
        print(f"\n正在加载 LoRA 适配器: {self.model_path}")
        self.model = PeftModel.from_pretrained(self.base_model, self.model_path)
        self.model.eval()
        print("✓ LoRA 适配器加载成功")
        
        # 5. 显示模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n模型信息:")
        print(f"  - 设备: {self.device}")
        if self.device == "cuda":
            if torch.cuda.is_available():
                print(f"  - PyTorch 可见的 GPU 数量: {torch.cuda.device_count()}")
                if self.gpu_id is not None:
                    print(f"  - 实际使用的物理 GPU ID: {self.gpu_id}（通过 CUDA_VISIBLE_DEVICES 设置）")
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    print(f"  - PyTorch GPU {i} (物理 GPU {self.gpu_id if self.gpu_id is not None else 'N/A'}): {gpu_name} ({gpu_memory:.2f} GB) [当前使用]")
        print(f"  - 总参数量: {total_params:,}")
        print(f"  - 模型路径: {self.model_path}")
        print("=" * 60)
        
        # 对话历史（使用 transformers 标准格式）
        self.conversation_history = []
        
        # 对话统计
        self.total_turns = 0
        self.start_time = None
        
        # 检查 tokenizer 是否支持 apply_chat_template
        if not hasattr(self.tokenizer, 'apply_chat_template') or self.tokenizer.chat_template is None:
            print("警告: tokenizer 不支持 apply_chat_template，将使用手动格式化")
            self.use_chat_template = False
        else:
            self.use_chat_template = True
            print("✓ 使用 tokenizer.apply_chat_template 进行对话格式化")
    
    def _setup_logging(self):
        """设置日志记录"""
        # 创建日志目录
        self.log_dir.mkdir(exist_ok=True)
        
        # 生成日志文件名（带时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"chat_{timestamp}.log"
        
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
            handlers=[file_handler, console_handler],
            force=True  # 强制重新配置，避免重复配置
        )
        
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file
        
        # 记录会话开始信息
        self.logger.info("=" * 60)
        self.logger.info("对话会话开始")
        self.logger.info("=" * 60)
        self.logger.info(f"模型路径: {self.model_path}")
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"日志文件: {log_file}")
        self.logger.info("=" * 60)
        
    def format_prompt(self, user_input, system_msg="You are a helpful assistant.", use_history=False):
        """
        格式化用户输入为对话格式（使用 transformers 的 apply_chat_template）
        
        Args:
            user_input: 用户输入
            system_msg: 系统消息
            use_history: 是否使用对话历史
        
        Returns:
            格式化后的提示文本
        """
        if self.use_chat_template:
            # 使用 transformers 标准的 apply_chat_template 方法
            messages = []
            
            # 添加系统消息
            if system_msg:
                messages.append({"role": "system", "content": system_msg})
            
            # 添加对话历史（最近3轮，避免上下文过长）
            if use_history and self.conversation_history:
                for item in self.conversation_history[-3:]:
                    messages.append({"role": "user", "content": item['user']})
                    messages.append({"role": "assistant", "content": item['assistant']})
            
            # 添加当前用户输入
            messages.append({"role": "user", "content": user_input})
            
            # 使用 apply_chat_template 格式化
            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return formatted_text
        else:
            # 回退到手动格式化（兼容旧版本）
            history_text = ""
            if use_history and self.conversation_history:
                for item in self.conversation_history[-3:]:
                    history_text += f"<|im_start|>user\n{item['user']}<|im_end|>\n"
                    history_text += f"<|im_start|>assistant\n{item['assistant']}<|im_end|>\n"
            
            current_user = f"<|im_start|>user\n{user_input}<|im_end|>\n"
            formatted_text = (
                f"<|im_start|>system\n{system_msg}<|im_end|>\n"
                f"{history_text}"
                f"{current_user}"
                f"<|im_start|>assistant\n"
            )
            return formatted_text
    
    def generate_response(self, user_input, temperature=0.7, top_p=0.8, max_new_tokens=256, use_history=False):
        """
        生成回复
        
        Args:
            user_input: 用户输入
            temperature: 温度参数（控制随机性）
            top_p: nucleus sampling 参数
            max_new_tokens: 最大生成 token 数
            use_history: 是否使用对话历史
        
        Returns:
            模型生成的回复
        """
        # 格式化提示
        prompt = self.format_prompt(user_input, use_history=use_history)
        
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # 生成回复
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,  # 增加重复惩罚，防止重复生成
                no_repeat_ngram_size=3,  # 禁止重复的 3-gram
                early_stopping=True,  # 提前停止
            )
        
        # 解码输出（只解码新生成的部分）
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]  # 只取新生成的部分
        assistant_response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 清理回复：移除可能的特殊标记和重复内容
        assistant_response = assistant_response.strip()
        
        # 如果使用 chat_template，通常不需要手动处理特殊标记
        # 但为了兼容性，仍然进行清理
        if "<|im_end|>" in assistant_response:
            assistant_response = assistant_response.split("<|im_end|>")[0].strip()
        
        # 移除其他可能的特殊标记
        assistant_response = assistant_response.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
        
        # 防止重复：检测并移除重复的句子或短语
        if len(assistant_response) > 10:
            # 按句子分割
            sentences = assistant_response.split('。')
            if len(sentences) > 1:
                # 移除重复的句子
                seen = set()
                unique_sentences = []
                for sent in sentences:
                    sent = sent.strip()
                    if sent and sent not in seen:
                        seen.add(sent)
                        unique_sentences.append(sent)
                assistant_response = '。'.join(unique_sentences)
            
            # 检查是否有明显的短语重复
            words = assistant_response.split()
            if len(words) > 8:
                # 检查是否有重复的短语（4-6个词）
                for n in range(4, min(7, len(words) // 2 + 1)):
                    for i in range(len(words) - n * 2):
                        phrase = " ".join(words[i:i+n])
                        remaining = " ".join(words[i+n:])
                        if phrase in remaining:
                            # 找到重复，只保留第一次出现
                            idx = remaining.find(phrase)
                            if idx != -1:
                                # 移除重复部分
                                before_repeat = " ".join(words[:i+n])
                                after_repeat = remaining[idx+n:].strip()
                                assistant_response = (before_repeat + " " + after_repeat).strip()
                                break
                    else:
                        continue
                    break
        
        return assistant_response
    
    def chat(self):
        """开始交互式对话"""
        self.start_time = time.time()
        
        print("\n" + "=" * 60)
        print("交互式对话开始！")
        print("=" * 60)
        print("提示:")
        print("  - 输入 'quit', 'exit', 'q' 或按 Ctrl+C 退出")
        print("  - 输入 'clear' 或 'reset' 清空对话历史")
        print("  - 输入 'help' 查看帮助信息")
        print(f"  - 对话日志: {self.log_file}")
        print("=" * 60)
        print()
        
        self.logger.info("交互式对话开始")
        self.logger.info("提示: 输入 'quit', 'exit', 'q' 退出，'clear' 清空历史，'help' 查看帮助")
        
        while True:
            try:
                # 获取用户输入
                user_input = input("您: ").strip()
                
                # 处理特殊命令
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    # 记录会话结束
                    if self.start_time:
                        duration = time.time() - self.start_time
                        self.logger.info("=" * 60)
                        self.logger.info("对话会话结束")
                        self.logger.info(f"总对话轮数: {self.total_turns}")
                        self.logger.info(f"会话时长: {duration/60:.2f} 分钟 ({duration:.2f} 秒)")
                        self.logger.info("=" * 60)
                    print("\n再见！")
                    break
                
                if user_input.lower() in ['clear', 'reset']:
                    self.conversation_history = []
                    self.logger.info("用户清空对话历史")
                    print("✓ 对话历史已清空\n")
                    continue
                
                if user_input.lower() == 'help':
                    help_msg = "用户查看帮助信息"
                    self.logger.info(help_msg)
                    print("\n可用命令:")
                    print("  - quit/exit/q: 退出对话")
                    print("  - clear/reset: 清空对话历史")
                    print("  - help: 显示此帮助信息")
                    print("  - history: 显示对话历史")
                    print()
                    continue
                
                if user_input.lower() == 'history':
                    self.logger.info(f"用户查看对话历史（共 {len(self.conversation_history)} 轮）")
                    if self.conversation_history:
                        print(f"\n对话历史（共 {len(self.conversation_history)} 轮）:")
                        print("-" * 60)
                        for i, item in enumerate(self.conversation_history[-10:], 1):  # 显示最近10轮
                            print(f"{i}. 您: {item['user']}")
                            print(f"   AI: {item['assistant'][:100]}..." if len(item['assistant']) > 100 else f"   AI: {item['assistant']}")
                        print("-" * 60)
                    else:
                        print("对话历史为空")
                    print()
                    continue
                
                # 记录用户输入
                self.logger.info(f"[用户] {user_input}")
                self.total_turns += 1
                
                # 生成回复
                print("AI: ", end="", flush=True)
                generation_start = time.time()
                
                # 使用对话历史（最近5轮）
                response = self.generate_response(user_input, use_history=True)
                
                generation_time = time.time() - generation_start
                
                print(response)
                print()
                
                # 记录AI回复
                self.logger.info(f"[AI] {response}")
                self.logger.info(f"生成耗时: {generation_time:.2f} 秒")
                self.logger.info("-" * 60)
                
                # 保存对话历史
                self.conversation_history.append({
                    "user": user_input,
                    "assistant": response,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "generation_time": generation_time
                })
                
            except KeyboardInterrupt:
                if self.start_time:
                    duration = time.time() - self.start_time
                    self.logger.info("=" * 60)
                    self.logger.info("对话会话中断（Ctrl+C）")
                    self.logger.info(f"总对话轮数: {self.total_turns}")
                    self.logger.info(f"会话时长: {duration/60:.2f} 分钟 ({duration:.2f} 秒)")
                    self.logger.info("=" * 60)
                print("\n\n再见！")
                break
            except Exception as e:
                error_msg = f"错误: {e}"
                self.logger.error(error_msg, exc_info=True)
                print(f"\n{error_msg}")
                print("请重试或输入 'quit' 退出\n")


def main():
    """主函数"""
    # 获取脚本所在目录，用于解析相对路径
    _SCRIPT_DIR = Path(__file__).parent.resolve()
    _PROJECT_ROOT = _SCRIPT_DIR.parent  # tool/ 的父目录
    _DEFAULT_MODEL_PATH = str(_PROJECT_ROOT / "output" / "qwen-1.5b-lora" / "run_20251120_191811")
    _DEFAULT_LOG_DIR = str(_PROJECT_ROOT / "logs")
    
    # 创建新的参数解析器（--gpu 参数已在文件顶部解析）
    parser = argparse.ArgumentParser(description="使用训练好的 LoRA 模型进行交互式对话")
    parser.add_argument(
        "--model_path",
        type=str,
        default=_DEFAULT_MODEL_PATH,
        help=f"LoRA 模型路径（默认: {_DEFAULT_MODEL_PATH}）"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="基础模型名称（默认: 从 adapter_config.json 读取）"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="设备类型（默认: 自动选择）"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="最大生成长度（默认: 2048）"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=_DEFAULT_LOG_DIR,
        help=f"日志目录（默认: {_DEFAULT_LOG_DIR}）"
    )
    # 注意：--gpu 参数已在文件顶部解析，这里不再重复添加
    
    args = parser.parse_args()
    
    # 验证 GPU 是否可用
    if torch.cuda.is_available():
        if torch.cuda.device_count() == 0:
            print(f"警告: 设置 CUDA_VISIBLE_DEVICES={args_pre.gpu} 后，没有可用的 GPU")
            print("请检查 GPU ID 是否正确")
        else:
            print(f"PyTorch 可见的 GPU 数量: {torch.cuda.device_count()}")
            print(f"实际使用的物理 GPU: {args_pre.gpu}（在 PyTorch 中映射为 cuda:0）")
    
    # 检查模型路径
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"错误: 模型路径不存在: {model_path}")
        print(f"请确保模型已训练完成，路径为: {model_path}")
        return
    
    if not (model_path / "adapter_config.json").exists():
        print(f"错误: 在 {model_path} 中未找到 adapter_config.json")
        print("请确保这是有效的 LoRA 模型目录")
        return
    
    # 创建对话机器人并开始对话
    try:
        chatbot = ChatBot(
            model_path=args.model_path,
            base_model_name=args.base_model,
            device=args.device,
            max_length=args.max_length,
            log_dir=args.log_dir,
            gpu_id=args_pre.gpu  # 传递 GPU ID
        )
        chatbot.chat()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

