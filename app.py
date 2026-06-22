# -*- coding: utf-8 -*-
"""
=============================================================================
English OCR 手写识别翻译助手 - Gradio 交互界面
=============================================================================
启动: python app.py  |  访问: http://localhost:7860
=============================================================================
"""

import os, sys, io, time, traceback
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import gradio as gr
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    GRADIO_TITLE, GRADIO_THEME, GRADIO_SERVER_PORT, GRADIO_SHARE,
    LLM_PROVIDERS, DEFAULT_PROVIDER, MODELS_DIR, EASYOCR_MODEL_DIR,
    HF_CACHE_DIR, API_KEY_FILE, load_api_key, save_api_key,
)
from modules.ocr_engine import OCREngine
from modules.knowledge_base import KnowledgeBase
from modules.llm_client import LLMClient

# ==================== 全局 ====================
_ocr_engine = None
_kb = None
_llm_client = None


def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = OCREngine()
    return _ocr_engine


def get_kb():
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
        _kb.load_data()
        _kb.build_index()
    return _kb


def get_llm_client(provider=None, api_key=None):
    global _llm_client
    key = api_key or load_api_key()
    _llm_client = LLMClient(provider=provider, api_key_override=key)
    return _llm_client


def check_api_key_status():
    key = load_api_key()
    if key and key != "your-api-key-here" and len(key) > 10:
        return f"已配置 (…{key[-8:]})"
    return "未配置"


def save_api_key_handler(key):
    if not key or not key.strip(): return "请输入 API Key"
    if len(key.strip()) < 10: return "Key 太短，格式不对"
    return f"已保存" if save_api_key(key.strip()) else "保存失败"


def switch_llm_provider(provider, api_key=""):
    try:
        key = api_key.strip() or load_api_key()
        llm = LLMClient(provider=provider, api_key_override=key)
        return llm.test_connection()["message"]
    except Exception as e:
        return f"连接失败: {str(e)[:100]}"


# ==================== 上传即 OCR + 翻译 ====================

def auto_ocr_translate(image, enable_preprocess, enable_deskew, llm_provider, api_key):
    """上传图片后自动 OCR 识别 + 翻译"""
    if image is None:
        return "", "", "", "请先上传图片"

    # Step 1: OCR
    engine = get_ocr_engine()
    result = engine.recognize(image, preprocess=enable_preprocess,
                               enable_deskew=enable_deskew, enable_border_removal=True)

    if result.get("error"):
        return "", f"OCR 失败: {result['error']}", "", ""

    text = result.get("text", "")
    keywords = result.get("keywords", [])
    keywords_str = ", ".join(keywords) if keywords else "(无)"

    if not text.strip():
        return "", "(未识别到文字)", "", keywords_str

    # Step 2: 翻译
    try:
        llm = get_llm_client(provider=llm_provider, api_key=api_key)
        from modules.prompts import build_prompt
        prompt = build_prompt("chinese_translation", text)
        resp = llm.chat(prompt["user"], prompt["system"])
        translation = resp["content"] if resp["success"] else f"翻译失败: {resp['error']}"
    except Exception as e:
        translation = f"翻译出错: {str(e)[:100]}"

    status = f"OCR 完成 | 识别词数: {len(result.get('raw_words', []))}"
    return text, translation, status, keywords_str


# ==================== AI 分析独立按钮 ====================

def do_analyze(ocr_text, keywords_str, llm_provider, api_key):
    """AI 多维度分析"""
    if not ocr_text or not ocr_text.strip():
        return "", "", "", "", "请先进行 OCR 识别"

    try:
        kb = get_kb()
        keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
        kb_results = kb.search_by_keywords(keywords) if keywords else kb.search(ocr_text)

        llm = get_llm_client(provider=llm_provider, api_key=api_key)
        results = llm.multi_task_analyze(ocr_text=ocr_text, kb_results=kb_results)

        grammar = results.get("grammar_correction", {}).get("content", "语法分析失败")
        sentence = results.get("sentence_analysis", {}).get("content", "句子分析失败")
        vocab = results.get("vocabulary_expansion", {}).get("content", "词汇分析失败")
        comprehensive = results.get("comprehensive", {}).get("content", "综合分析失败")

        status = f"分析完成 | 知识库({kb.search_mode}): 匹配 {len(kb_results) if kb_results else 0} 条"
        return grammar, sentence, vocab, comprehensive, status

    except ValueError as e:
        err = f"API 配置错误 — 请在设置中填入 API Key"
        return err, err, err, err, err
    except Exception as e:
        err = f"分析失败: {str(e)[:200]}"
        return err, err, err, err, err


# ==================== UI ====================

def create_ui():
    with gr.Blocks(title=GRADIO_TITLE) as demo:

        gr.Markdown("""
        # 英语手写 OCR 识别翻译助手
        ### 上传图片 → OCR 识别 → 翻译 → AI 深度分析
        """)

        # ---- 设置 ----
        with gr.Accordion("⚙️ 设置", open=False):
            with gr.Row():
                with gr.Column(scale=3):
                    api_key_input = gr.Textbox(
                        label="API Key",
                        placeholder="输入后点保存，下次自动加载",
                        value=load_api_key() if load_api_key() != "your-api-key-here" else "",
                        info=f"保存在 {API_KEY_FILE.name}",
                    )
                with gr.Column(scale=1):
                    save_key_btn = gr.Button("💾 保存", size="sm")
                    api_key_status = gr.Textbox(value=check_api_key_status(), interactive=False, lines=1)
            with gr.Row():
                enable_preprocess = gr.Checkbox(value=True, label="图像预处理")
                enable_deskew = gr.Checkbox(value=True, label="倾斜矫正")
                llm_provider = gr.Dropdown(
                    choices=list(LLM_PROVIDERS.keys()), value=DEFAULT_PROVIDER,
                    label="大模型", info="DeepSeek 推荐"
                )
                test_btn = gr.Button("🔌 测试连接", size="sm")
            connection_status = gr.Textbox(label="连接测试", lines=1, interactive=False)

            save_key_btn.click(save_api_key_handler, [api_key_input], [api_key_status])
            test_btn.click(switch_llm_provider, [llm_provider, api_key_input], [connection_status])

        # ---- 主体 ----
        with gr.Row():
            with gr.Column(scale=2):
                image_input = gr.Image(label="手写图片（上传即自动识别+翻译）", type="numpy", sources=["upload", "clipboard"])
                status_display = gr.Textbox(label="状态", interactive=False, lines=1)

            with gr.Column(scale=4):
                ocr_text_display = gr.Textbox(
                    label="识别文本（可手动编辑）", lines=6, interactive=True,
                    placeholder="上传图片后自动识别..."
                )
                translation_display = gr.Textbox(
                    label="中文翻译", lines=6, interactive=False,
                    placeholder="自动翻译..."
                )

        # ---- AI 分析 ----
        gr.Markdown("---")
        with gr.Row():
            analyze_btn = gr.Button("🤖 AI 深度分析", variant="primary", size="lg")
            keywords_display = gr.Textbox(label="关键词 (自动提取)", interactive=False, visible=False)

        with gr.Tabs():
            with gr.TabItem("📝 语法纠错"):
                grammar_display = gr.Textbox(label="", lines=10, interactive=False)
            with gr.TabItem("🔍 长难句分析"):
                sentence_display = gr.Textbox(label="", lines=10, interactive=False)
            with gr.TabItem("📖 词汇拓展"):
                vocab_display = gr.Textbox(label="", lines=10, interactive=False)
            with gr.TabItem("📋 综合分析"):
                comp_display = gr.Textbox(label="", lines=12, interactive=False)

        # ---- 事件绑定 ----
        image_input.change(
            auto_ocr_translate,
            [image_input, enable_preprocess, enable_deskew, llm_provider, api_key_input],
            [ocr_text_display, translation_display, status_display, keywords_display],
        )
        analyze_btn.click(
            do_analyze,
            [ocr_text_display, keywords_display, llm_provider, api_key_input],
            [grammar_display, sentence_display, vocab_display, comp_display, status_display],
        )

        # ---- 说明 ----
        with gr.Accordion("📖 使用说明", open=False):
            gr.Markdown("""
            ### 三步操作
            1. **上传图片** → 点「OCR 识别」
            2. 确认识别文本无误后 → 点「翻译」
            3. 点「AI 深度分析」→ 查看语法/句法/词汇/综合分析

            ### Tips
            - 识别文本可以手动编辑修正后再翻译/分析
            - DeepSeek API Key 在 platform.deepseek.com 免费获取
            - 整个文件夹拷到任何电脑都能用
            """)

    return demo


# ==================== 启动 ====================

if __name__ == "__main__":
    print("=" * 50)
    print(f"English OCR Assistant")
    print(f"  Models: {MODELS_DIR}")
    print(f"  Access: http://localhost:{GRADIO_SERVER_PORT}")
    print("=" * 50)

    import os as _os
    _os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

    demo = create_ui()
    demo.launch(server_name="127.0.0.1", server_port=GRADIO_SERVER_PORT, share=True)
