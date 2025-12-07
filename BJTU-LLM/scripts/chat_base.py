#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用基础 Qwen/Qwen2.5-1.5B-Instruct 模型进行交互式对话

使用方法:
    python chat_base_model.py
    python chat_base_model.py --model_name Qwen/Qwen2.5-1.5B-Instruct
"""

import os
# 配置 HuggingFace 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_CACHE'] = os.environ.get('HUGGINGFACE_HUB_CACHE', os.path.expanduser('~/.cache/huggingface'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from pathlib import Path
import logging
from datetime import datetime
import time


class BaseChatBot:
    """使用基础模型的交互式对话机器人"""
    
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", device=None, max_length=2048, log_dir="logs"):
        """
        初始化对话机器人
        
        Args:
            model_name: 基础模型名称
            device: 设备（cuda/cpu），如果为 None 自动选择
            max_length: 最大生成长度
            log_dir: 日志目录
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.log_dir = Path(log_dir)
        
        # 设置日志
        self._setup_logging()
        
        print("=" * 60)
        print("正在加载基础模型...")
        print("=" * 60)
        print(f"模型: {self.model_name}")
        print(f"设备: {self.device}")
        print()
        
        # 1. 加载 tokenizer
        print("正在加载 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=False
        )
        
        # 设置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print(f"✓ Tokenizer 加载成功，词汇表大小: {len(self.tokenizer)}")
        self.logger.info(f"Tokenizer 加载成功，词汇表大小: {len(self.tokenizer)}")
        
        # 2. 加载模型
        print("\n正在加载模型...")
        self.logger.info("正在加载基础模型...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            local_files_only=False
        )
        self.model.eval()
        
        print("✓ 模型加载成功")
        
        # 3. 显示模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n模型信息:")
        print(f"  - 设备: {self.device}")
        print(f"  - 总参数量: {total_params:,}")
        print(f"  - 模型名称: {self.model_name}")
        print("=" * 60)
        
        self.logger.info(f"模型加载成功")
        self.logger.info(f"  - 设备: {self.device}")
        self.logger.info(f"  - 总参数量: {total_params:,}")
        self.logger.info(f"  - 模型名称: {self.model_name}")
        
        # 对话历史
        self.conversation_history = []
        
        # 对话统计
        self.total_turns = 0
        self.start_time = None
    
    def _setup_logging(self):
        """设置日志记录"""
        # 创建日志目录
        self.log_dir.mkdir(exist_ok=True)
        
        # 生成日志文件名（带时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"chat_base_{timestamp}.log"
        
        # 配置日志格式
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        
        # 配置日志：同时输出到文件和控制台
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            datefmt=date_format,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ],
            force=True  # 强制重新配置，避免重复配置
        )
        
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file
        
        # 记录会话开始信息
        self.logger.info("=" * 60)
        self.logger.info("对话会话开始（使用基础模型）")
        self.logger.info("=" * 60)
        self.logger.info(f"模型名称: {self.model_name}")
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"日志文件: {log_file}")
        self.logger.info("=" * 60)
        
    def format_prompt(self, user_input, system_msg="You are a helpful assistant.", use_history=False):
        """
        格式化用户输入为 Qwen2.5-Instruct 对话格式
        
        Args:
            user_input: 用户输入
            system_msg: 系统消息
            use_history: 是否使用对话历史
        
        Returns:
            格式化后的提示文本
        """
        # 构建对话历史
        history_text = ""
        if use_history and self.conversation_history:
            # 只使用最近3轮对话，避免上下文过长导致重复
            for item in self.conversation_history[-3:]:  # 减少到3轮对话
                history_text += f"<|im_start|>user\n{item['user']}<|im_end|>\n"
                history_text += f"<|im_start|>assistant\n{item['assistant']}<|im_end|>\n"
        
        # 当前用户输入
        current_user = f"<|im_start|>user\n{user_input}<|im_end|>\n"
        
        # 组合完整提示
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
        
        # 如果包含结束标记，只取标记之前的内容
        if "<|im_end|>" in assistant_response:
            assistant_response = assistant_response.split("<|im_end|>")[0].strip()
        
        # 如果包含其他特殊标记，也移除
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
                
                # 使用对话历史（最近3轮）
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
    _DEFAULT_LOG_DIR = str(_PROJECT_ROOT / "logs")
    
    parser = argparse.ArgumentParser(description="使用基础 Qwen 模型进行交互式对话")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="模型名称（默认: Qwen/Qwen2.5-1.5B-Instruct）"
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
    
    args = parser.parse_args()
    
    # 创建对话机器人并开始对话
    try:
        chatbot = BaseChatBot(
            model_name=args.model_name,
            device=args.device,
            max_length=args.max_length,
            log_dir=args.log_dir
        )
        chatbot.chat()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

