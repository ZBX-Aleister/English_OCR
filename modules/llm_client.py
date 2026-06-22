# -*- coding: utf-8 -*-
"""
=============================================================================
大模型客户端模块 - 统一 LLM API 调用接口
=============================================================================
功能: 封装 OpenAI 兼容 API，支持通义千问 / GLM / OpenAI 多后端切换

核心类:
  LLMClient: 初始化客户端，提供 chat() / multi_task_analyze() 等方法

支持的提供商:
  - 通义千问 (Qwen):   dashscope.aliyuncs.com
  - 智谱 GLM:           open.bigmodel.cn
  - OpenAI GPT:         api.openai.com
  - 任何兼容 OpenAI API 格式的服务

并发策略:
  - multi_task_analyze 使用线程池并发调用多个任务
  - 最大并发数 MAX_CONCURRENT_TASKS
=============================================================================
"""

import time
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from config import (
    LLM_API_KEY,
    LLM_PROVIDERS,
    DEFAULT_PROVIDER,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    LLM_MAX_RETRIES,
)
from modules.prompts import build_prompt, build_comprehensive_prompt

# 最大并发任务数
MAX_CONCURRENT_TASKS = 3


class LLMClient:
    """
    大模型 API 客户端

    用法:
        client = LLMClient(provider="qwen")
        result = client.chat("Hello, translate this.")

        # 多任务并发分析
        results = client.multi_task_analyze(
            ocr_text="The quick brown fox",
            tasks=["chinese_translation", "grammar_correction"]
        )
    """

    def __init__(self, provider: str = None, api_key_override: str = None):
        """
        初始化 LLM 客户端

        Args:
            provider: 提供商标识 (deepseek / qwen / glm / openai)
                      留空使用 DEFAULT_PROVIDER
            api_key_override: 手动传入 API Key（优先级高于 config）
        """
        self.provider = provider or DEFAULT_PROVIDER

        if self.provider not in LLM_PROVIDERS:
            raise ValueError(
                f"不支持的提供商: {self.provider}\n"
                f"可用选项: {list(LLM_PROVIDERS.keys())}"
            )

        self._config = LLM_PROVIDERS[self.provider]

        # API Key: 参数传入 > 环境变量 > config 配置
        api_key = api_key_override or LLM_API_KEY
        if not api_key or api_key == "your-api-key-here":
            raise ValueError(
                "请在设置面板中填写 API Key 并点击保存\n"
                "或设置环境变量: set LLM_API_KEY=your-key\n"
                f"当前提供商: {self._config['name']}\n"
                f"获取 API Key: DeepSeek → platform.deepseek.com"
            )

        self._client = OpenAI(
            api_key=api_key,
            base_url=self._config["base_url"],
            timeout=LLM_TIMEOUT,
        )

        print(f"[LLM] 客户端初始化完成: {self._config['name']} ({self._config['model']})")

    @property
    def model_name(self) -> str:
        return self._config["model"]

    @property
    def provider_name(self) -> str:
        return self._config["name"]

    def chat(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> Dict:
        """
        单轮对话调用

        Args:
            user_prompt: 用户消息
            system_prompt: 系统提示词
            temperature: 温度参数 (0-2)
            max_tokens: 最大输出 token 数

        Returns:
            {
                "success": bool,
                "content": str,         # LLM 回复文本
                "model": str,           # 实际使用的模型
                "usage": {              # token 用量
                    "prompt_tokens": int,
                    "completion_tokens": int,
                    "total_tokens": int,
                },
                "latency": float,       # 响应延迟（秒）
                "error": str or None,
            }
        """
        result = {
            "success": False,
            "content": "",
            "model": self._config["model"],
            "usage": {},
            "latency": 0.0,
            "error": None,
        }

        last_error = None
        for attempt in range(LLM_MAX_RETRIES):
            try:
                start_time = time.time()

                response = self._client.chat.completions.create(
                    model=self._config["model"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                elapsed = time.time() - start_time

                result["success"] = True
                result["content"] = response.choices[0].message.content
                result["latency"] = round(elapsed, 3)
                result["model"] = response.model

                if hasattr(response, "usage") and response.usage:
                    result["usage"] = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }

                return result

            except Exception as e:
                last_error = str(e)
                if attempt < LLM_MAX_RETRIES - 1:
                    wait_time = 2 ** attempt  # 指数退避: 1s, 2s, 4s
                    print(f"[LLM] 第 {attempt + 1} 次调用失败，{wait_time}s 后重试: {e}")
                    time.sleep(wait_time)

        result["error"] = f"调用失败（已重试 {LLM_MAX_RETRIES} 次）: {last_error}"
        return result

    def analyze_single(
        self,
        task_type: str,
        ocr_text: str,
        kb_context: str = "",
    ) -> Dict:
        """
        执行单任务分析

        Args:
            task_type: 任务类型 (chinese_translation / grammar_correction / ...)
            ocr_text: OCR 识别文本
            kb_context: 知识库上下文

        Returns:
            {
                "task_type": str,
                "success": bool,
                "content": str,
                "error": str or None,
                "latency": float,
                "usage": dict,
            }
        """
        prompt_data = build_prompt(task_type, ocr_text, kb_context)
        response = self.chat(
            user_prompt=prompt_data["user"],
            system_prompt=prompt_data["system"],
        )

        return {
            "task_type": task_type,
            "success": response["success"],
            "content": response["content"],
            "error": response["error"],
            "latency": response["latency"],
            "usage": response.get("usage", {}),
        }

    def multi_task_analyze(
        self,
        ocr_text: str,
        kb_results: Optional[List[Dict]] = None,
        tasks: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """
        多任务并发分析 - 同时执行多种分析任务

        Args:
            ocr_text: OCR 识别文本
            kb_results: 知识库检索结果（用于 comprehensive 任务）
            tasks: 任务列表，默认为全部 5 种

        Returns:
            {
                "chinese_translation": { ... },
                "grammar_correction": { ... },
                "sentence_analysis": { ... },
                "vocabulary_expansion": { ... },
                "comprehensive": { ... },
            }
        """
        if tasks is None:
            tasks = [
                "chinese_translation",
                "grammar_correction",
                "sentence_analysis",
                "vocabulary_expansion",
                "comprehensive",
            ]

        # 为 comprehensive 任务准备知识库上下文
        kb_context = ""
        if kb_results:
            from modules.prompts import build_comprehensive_prompt
            comp_prompt = build_comprehensive_prompt(ocr_text, kb_results)
            kb_context = comp_prompt["user"]

        results = {}

        # 使用线程池并发执行
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TASKS) as executor:
            future_map = {}

            for task_type in tasks:
                if task_type == "comprehensive" and kb_results:
                    # comprehensive 使用专用 prompt 构建
                    comp_prompt = build_comprehensive_prompt(ocr_text, kb_results)
                    future = executor.submit(
                        self.chat,
                        user_prompt=comp_prompt["user"],
                        system_prompt=comp_prompt["system"],
                    )
                else:
                    prompt_data = build_prompt(task_type, ocr_text)
                    future = executor.submit(
                        self.chat,
                        user_prompt=prompt_data["user"],
                        system_prompt=prompt_data["system"],
                    )

                future_map[future] = task_type

            for future in as_completed(future_map):
                task_type = future_map[future]
                try:
                    response = future.result()
                    results[task_type] = {
                        "task_type": task_type,
                        "success": response["success"],
                        "content": response["content"],
                        "error": response["error"],
                        "latency": response["latency"],
                        "usage": response.get("usage", {}),
                    }
                except Exception as e:
                    results[task_type] = {
                        "task_type": task_type,
                        "success": False,
                        "content": "",
                        "error": str(e),
                        "latency": 0,
                        "usage": {},
                    }

        return results

    def test_connection(self) -> Dict:
        """
        测试 API 连接是否正常

        Returns:
            {"success": bool, "message": str, "latency": float}
        """
        try:
            start = time.time()
            response = self._client.chat.completions.create(
                model=self._config["model"],
                messages=[{"role": "user", "content": "Hello, respond with just 'OK'."}],
                max_tokens=10,
            )
            elapsed = time.time() - start

            return {
                "success": True,
                "message": f"✅ 连接成功！提供商: {self._config['name']}, 模型: {self._config['model']}, 延迟: {elapsed:.2f}s",
                "latency": round(elapsed, 3),
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ 连接失败: {str(e)}\n请检查 API Key 和网络连接",
                "latency": 0,
            }
