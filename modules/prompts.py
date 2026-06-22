# -*- coding: utf-8 -*-
"""
=============================================================================
分层提示词模块 - 多任务 Prompt 模板库
=============================================================================
功能: 管理 5 种任务的提示词模板，支持自动拼接 OCR 文本 + 知识库内容

5 种分析任务:
  1. chinese_translation   - 中文翻译（自然流畅）
  2. grammar_correction    - 语法纠错（逐句分析）
  3. sentence_analysis     - 长难句拆分（结构解析）
  4. vocabulary_expansion  - 单词拓展（释义+同义词+例句）
  5. comprehensive         - 综合分析（融合以上全部 + KB 内容）

每类 Prompt 分为:
  - system_prompt: 角色设定（告诉 LLM 它是谁、做什么）
  - user_template: 用户查询模板（拼接 OCR 文本和 KB 内容）
=============================================================================
"""

# ==================== 系统角色提示词 ====================

SYSTEM_PROMPTS = {
    "chinese_translation": """\
你是一位专业的英中翻译专家。你的任务是将英文文本翻译成自然流畅的中文。

翻译要求：
1. 准确传达原文意思，不添加不遗漏
2. 中文表达自然地道，符合中文阅读习惯
3. 对于学术/考试类文本，使用规范的书面语
4. 如果原文有语法错误，先按原意翻译，然后在译文后标注"（原文可能存在语法问题）"
5. 如果原文是手写OCR识别结果，可能存在拼写错误——请根据上下文合理推测原意后进行翻译

请直接输出翻译结果，不要添加额外的解释说明。""",

    "grammar_correction": """\
你是一位专业的英语语法教师，擅长发现和纠正英语写作中的语法错误。

你的任务是分析英文文本中的语法、拼写、标点、用词问题。

分析要求：
1. 逐句检查，标注所有语法错误
2. 对每个错误给出：
   - 错误位置（引用原句）
   - 错误类型（时态/语态/主谓一致/冠词/介词/拼写等）
   - 修正建议
   - 简要解释
3. 在最后给出修正后的完整文本
4. 如果文本无误，请明确说明"未发现语法错误"

格式要求：使用清晰的分段格式，便于阅读。""",

    "sentence_analysis": """\
你是一位英语语言学分析专家，擅长解析英语句子的语法结构。

你的任务是分析英文文本的句子结构，特别是长难句。

分析要求：
1. 识别每个句子的类型（简单句/并列句/复合句/并列复合句）
2. 拆解句子成分：
   - 主句和从句的划分
   - 主语、谓语、宾语、定语、状语等成分标注
   - 从句类型（定语从句/状语从句/名词性从句等）
3. 指出句子的主干（去掉修饰成分后的核心意思）
4. 对于长难句，给出"简化理解"版本

格式要求：使用树形或缩进结构展示句子分析，层次分明。""",

    "vocabulary_expansion": """\
你是一位英语词汇教学专家，擅长讲解单词用法和拓展词汇。

你的任务是分析文本中的重点词汇，提供详细的词汇学习资料。

分析要求：
1. 识别文本中的重点/高频词汇（如考研词汇、四六级高频词）
2. 对每个重点词汇提供：
   - 音标
   - 中文释义（多个义项）
   - 词性
   - 2-3 个同义词及其区别
   - 2 个实用例句
   - 常见搭配（collocations）
3. 优先分析实义词（名词、动词、形容词、副词）
4. 如果文本较短，分析所有词汇；较长则挑选 5-8 个最重要的

格式要求：每个词汇独立成段，使用清晰的结构。""",

    "comprehensive": """\
你是一位全能型英语学习助手。你将收到一段英文文本（可能来自手写OCR识别），以及从知识库中检索到的相关学习资料。

你需要从以下多个维度进行综合分析：
1. **中文翻译** - 将文本翻译成自然流畅的中文
2. **语法纠错** - 检查并纠正语法、拼写错误
3. **长难句分析** - 解析句子结构（如有长难句）
4. **词汇拓展** - 讲解重点词汇（释义、同义词、例句）

输出格式要求：
- 使用清晰的标题分隔各板块
- 每个板块用中文说明，英文例句保留原文
- 如果OCR文本存在明显识别错误，在翻译前先进行合理推测和修正
- 参考知识库内容，如发现文本中的词汇/语法点与知识库匹配，优先使用知识库中的讲解

请保持专业、详尽、教学化的风格。""",
}

# ==================== 用户查询模板 ====================

USER_TEMPLATES = {
    "chinese_translation": """\
请将以下英文文本翻译成中文：

{ocr_text}""",

    "grammar_correction": """\
请检查以下英文文本的语法错误并给出修正建议：

{ocr_text}""",

    "sentence_analysis": """\
请分析以下英文文本的句子结构：

{ocr_text}""",

    "vocabulary_expansion": """\
请分析以下英文文本中的重点词汇，提供词汇学习资料：

{ocr_text}""",

    "comprehensive": """\
请从翻译、语法、句子结构、词汇四个方面综合分析以下英文文本。

【待分析文本】
{ocr_text}

【知识库参考资料】
{kb_context}""",
}


# ==================== Prompt 构建函数 ====================

def build_prompt(task_type: str, ocr_text: str, kb_context: str = "") -> dict:
    """
    构建完整的 LLM 对话 Prompt

    Args:
        task_type: 任务类型 (chinese_translation / grammar_correction /
                   sentence_analysis / vocabulary_expansion / comprehensive)
        ocr_text: OCR 识别并清洗后的英文文本
        kb_context: 知识库检索结果文本（可选，用于 comprehensive 等任务）

    Returns:
        {
            "system": str,      # 系统提示词
            "user": str,        # 用户消息
            "messages": list,   # 完整的 messages 列表（可直接发送给 API）
        }
    """
    system_prompt = SYSTEM_PROMPTS.get(task_type, SYSTEM_PROMPTS["comprehensive"])
    user_template = USER_TEMPLATES.get(task_type, USER_TEMPLATES["comprehensive"])

    user_message = user_template.format(
        ocr_text=ocr_text,
        kb_context=kb_context if kb_context else "（无相关知识库内容）",
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    return {
        "system": system_prompt,
        "user": user_message,
        "messages": messages,
    }


def build_comprehensive_prompt(ocr_text: str, kb_results: list) -> dict:
    """
    构建综合分析 Prompt（融合知识库检索结果）

    将 JSON 格式的知识库检索结果格式化为可读文本，
    嵌入 comprehensive 提示词模板中。

    Args:
        ocr_text: OCR 识别的英文文本
        kb_results: 知识库检索结果列表 [{type, text, metadata, score}, ...]

    Returns:
        同 build_prompt 的返回格式
    """
    # 格式化知识库内容
    kb_parts = []
    if kb_results:
        for i, item in enumerate(kb_results, 1):
            item_type = item.get("type", "unknown")
            item_text = item.get("text", "")
            item_meta = item.get("metadata", {})
            score = item.get("score", 0)

            type_labels = {
                "vocab": "📖 词汇",
                "grammar": "📝 语法",
                "example": "📋 例句",
            }
            label = type_labels.get(item_type, "📌 其他")

            kb_parts.append(f"{label} #{i} (相关度: {score:.2f}): {item_text}")
            if item_meta:
                meta_str = ", ".join(f"{k}={v}" for k, v in item_meta.items())
                kb_parts.append(f"  附加信息: {meta_str}")

    kb_context = "\n".join(kb_parts) if kb_parts else "（无相关知识库匹配结果）"

    return build_prompt("comprehensive", ocr_text, kb_context)


# ==================== 用于对比实验的备选 Prompt ====================
# 在实际调优中，可以创建多个版本的 Prompt 并对比效果

PROMPT_VARIANTS = {
    "translation_v1": {
        "system": "你是英文翻译专家。请将以下英文翻译为中文，要求准确流畅。",
        "description": "简洁版 - 适用于短文本快速翻译",
    },
    "translation_v2": {
        "system": SYSTEM_PROMPTS["chinese_translation"],
        "description": "详细版 - 适用于学术/考试文本翻译（当前使用）",
    },
    "grammar_v1": {
        "system": "你是一名英语老师。找出以下文本中的所有语法错误，并给出正确版本。",
        "description": "简洁版 - 快速语法检查",
    },
    "grammar_v2": {
        "system": SYSTEM_PROMPTS["grammar_correction"],
        "description": "详细版 - 逐句分析+解释（当前使用）",
    },
}
