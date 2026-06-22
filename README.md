# 📝 英语手写 OCR 识别翻译助手

> **人工智能导论 课程设计（大作业）**
>
> 基于 EasyOCR + FAISS + 大语言模型的英语手写识别与智能学习系统

---

## 一、项目概述

本项目构建了一个完整的**英语手写 OCR 识别翻译助手**，能够：

1. 📷 **手写英文 OCR 识别** — 对手机拍照/扫描的手写英文图片进行预处理和文字识别
2. 📚 **本地知识库检索** — 基于 FAISS 向量检索的考研/四六级词汇语法库
3. 🤖 **大模型多维分析** — 调用 LLM API 进行中文翻译、语法纠错、长难句分析、词汇拓展
4. 🌐 **Gradio 可视化交互** — 一键上传图片，自动完成全流程分析

### 系统架构图（文字描述，方便转 PPT）

```
┌──────────────────────────────────────────────────────────────────┐
│                    英语手写 OCR 识别翻译助手                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌────────────┐ │
│  │ 图片上传  │───▶│ 图像预处理│───▶│ EasyOCR  │───▶│  文本清洗   │ │
│  │ (Gradio) │    │ (OpenCV) │    │  识别    │    │ (规则+正则)│ │
│  └──────────┘    └──────────┘    └──────────┘    └─────┬──────┘ │
│                                                        │        │
│                          ┌─────────────────────────────┘        │
│                          ▼                                      │
│                  ┌───────────────┐                               │
│                  │  关键词提取    │                               │
│                  └───────┬───────┘                               │
│                          │                                      │
│              ┌───────────┼───────────┐                          │
│              ▼                       ▼                          │
│     ┌──────────────┐       ┌──────────────┐                     │
│     │ FAISS 向量检索│       │ 提示词拼接    │                     │
│     │ (词汇+语法+例句)│      │ (分层模板)   │                     │
│     └──────┬───────┘       └──────┬───────┘                     │
│            │                      │                             │
│            └──────────┬───────────┘                             │
│                       ▼                                         │
│              ┌────────────────┐                                 │
│              │  大模型 API     │                                 │
│              │ (Qwen/GLM/GPT) │                                 │
│              └───────┬────────┘                                 │
│                      │                                          │
│       ┌──────────────┼──────────────┬──────────────┐           │
│       ▼              ▼              ▼              ▼           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────┐       │
│  │中文翻译  │  │语法纠错  │  │句子分析  │  │  词汇拓展    │       │
│  └─────────┘  └─────────┘  └─────────┘  └─────────────┘       │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                Gradio 前端展示 (标签页)                     │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 二、运行环境清单

### 2.1 硬件要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| CPU | Intel i3 / AMD R3 | Intel i5+ / AMD R5+ |
| 内存 | 8GB RAM | 16GB RAM |
| 硬盘 | 5GB 可用空间 | 10GB SSD |
| GPU | 无要求 (CPU 可运行) | NVIDIA GPU 4GB+ VRAM |
| 网络 | 需要联网 (调用 LLM API) | 稳定宽带 |

### 2.2 软件环境

| 软件 | 版本 | 说明 |
|------|------|------|
| Windows | 10/11 | 或 macOS/Linux |
| Python | 3.9 ~ 3.11 | **不要使用 3.12+**（FAISS 兼容性问题） |
| pip | 最新版 | `python -m pip install --upgrade pip` |
| Git | 2.x | 用于 Hugging Face 部署 |

### 2.3 Python 虚拟环境（推荐）

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境 (Windows)
venv\Scripts\activate

# 激活虚拟环境 (macOS/Linux)
source venv/bin/activate
```

---

## 三、快速启动（5 分钟）

### 3.1 安装依赖

```bash
# 进入项目目录
cd "English OCR"

# 安装所有依赖
pip install -r requirements.txt
```

### 3.2 配置 API Key

**方式一：环境变量（推荐）**
```bash
# Windows CMD
set LLM_API_KEY=your-api-key-here

# Windows PowerShell
$env:LLM_API_KEY="your-api-key-here"

# macOS/Linux
export LLM_API_KEY="your-api-key-here"
```

**方式二：直接修改 config.py**
```python
# 编辑 config.py 第 23 行
LLM_API_KEY = "your-api-key-here"
```

### 3.3 构建知识库（首次运行）

```bash
python knowledge/build_kb.py
```

输出示例：
```
==========================================
📚 English OCR 知识库构建工具
==========================================
[1/3] 加载 JSON 数据文件...
[KB] 加载词汇: 180 条
[KB] 加载语法错误: 20 条
[KB] 加载例句: 10 条
[KB] 数据加载完成，共 210 条记录
[2/3] 构建 FAISS 向量索引...
[KB] 正在编码 210 条记录...
[KB] FAISS 索引构建完成 (维度: 384, 条目数: 210)
[3/3] 验证检索功能...
✅ 知识库构建完成！
```

### 3.4 启动系统

```bash
python app.py
```

浏览器访问: **http://localhost:7860**

---

## 四、分模块技术详解

### 4.1 图像预处理模块 (`modules/preprocessing.py`)

| 步骤 | 技术 | 解决什么问题 |
|------|------|------------|
| 灰度化 | `cv2.cvtColor` | 减少颜色噪声，简化后续处理 |
| 降噪 | `GaussianBlur` + `fastNlMeansDenoising` | 去除拍照噪点、纸张纹理 |
| 对比度增强 | CLAHE 自适应直方图均衡 | 使浅色铅笔字迹更清晰 |
| 二值化 | 自适应高斯阈值 | 转为纯黑白，突出文字轮廓 |
| 倾斜矫正 | 投影轮廓方差法 | 修正拍照角度偏差（±45°） |
| 裁剪 | 轮廓检测 + 边界框 | 去除白边，聚焦文字区域 |

### 4.2 OCR 引擎 (`modules/ocr_engine.py`)

- **工具**: EasyOCR (Jaided AI)
- **语言模型**: English (`en`)
- **置信度过滤**: 默认阈值 0.3，低置信度结果丢弃
- **GPU 支持**: 自动检测 GPU，失败则回退 CPU
- **文本清洗**: 去乱码、修正 OCR 常见混淆（rn→m, cl→d 等）

### 4.3 大模型客户端 (`modules/llm_client.py`)

- **协议**: OpenAI 兼容 API（一个接口兼容所有后端）
- **支持的提供商**:
  - 通义千问 (Qwen-Turbo) — dashscope.aliyuncs.com
  - 智谱 GLM (GLM-4-Flash) — open.bigmodel.cn
  - OpenAI (GPT-4o-mini) — api.openai.com
- **并发策略**: 线程池 3 并发，指数退避重试

### 4.4 提示词模板 (`modules/prompts.py`)

5 种专业提示词，每种包含 system prompt + user template：

1. **中文翻译** — 翻译专家角色，要求准确流畅
2. **语法纠错** — 英语教师角色，逐句分析标注
3. **长难句分析** — 语言学专家角色，成分拆解
4. **词汇拓展** — 词汇教学角色，释义+同义+搭配
5. **综合分析** — 融合以上 + 知识库内容

### 4.5 知识库 (`modules/knowledge_base.py`)

| 组件 | 技术 | 说明 |
|------|------|------|
| 数据存储 | JSON 文件 | 词汇(~180条)、语法(~20条)、例句(~10条) |
| 嵌入模型 | all-MiniLM-L6-v2 | 384 维向量，模型约 80MB |
| 向量索引 | FAISS IndexFlatL2 | L2 距离搜索，归一化后等价余弦相似度 |
| 搜索策略 | 语义 + 关键词混合 | 先提取关键词，再向量检索 |

---

## 五、搭建总流程

```
Phase 1: 环境搭建
  ├── 安装 Python 3.9-3.11
  ├── 创建虚拟环境
  ├── pip install -r requirements.txt
  └── 配置 LLM_API_KEY

Phase 2: 知识库构建
  ├── 导入 vocabulary.json (考研/四六级词汇)
  ├── 导入 grammar_errors.json (常见语法错误)
  ├── 导入 examples.json (标准例句)
  └── python knowledge/build_kb.py → 生成 FAISS 索引

Phase 3: 模块测试
  ├── 测试 OCR 识别: python -c "from modules.ocr_engine import OCREngine; ..."
  ├── 测试 LLM 连接: python -c "from modules.llm_client import LLMClient; ..."
  └── 测试知识库:  python -c "from modules.knowledge_base import KnowledgeBase; ..."

Phase 4: 前端启动
  └── python app.py → http://localhost:7860

Phase 5: 全模块联调
  ├── 上传图片 → 检查预处理效果
  ├── 验证 OCR 文本 → 手动修正后重新分析
  ├── 检查知识库匹配结果 → 如有缺失补充数据
  └── 对比不同 LLM 的分析质量

Phase 6: 迭代优化
  ├── 调整预处理参数（config.py）
  ├── 优化提示词（modules/prompts.py）
  ├── 扩充知识库（添加更多 JSON 条目）
  └── 测试不同置信度阈值

Phase 7: 线上部署
  ├── 创建 Hugging Face Space
  ├── 设置 API Key Secret
  ├── git push → 自动构建
  └── 获取公开访问链接
```

---

## 六、全模块联调说明

### 完整数据流转

```
用户上传手写图片
    │
    ▼
图像预处理（灰度化→降噪→对比度增强→二值化→倾斜矫正→裁剪）
    │
    ▼
EasyOCR 识别 → 置信度过滤 → 文本清洗 → 提取关键词
    │                                │
    │                                ▼
    │                          FAISS 知识库检索
    │                            （词汇+语法+例句）
    │                                │
    └──────────┬─────────────────────┘
               ▼
         提示词拼接（OCR 文本 + KB 内容 + 任务指令）
               │
               ▼
         大模型 API 调用（5 任务并发）
               │
    ┌──────────┼──────────┬──────────┐
    ▼          ▼          ▼          ▼
  翻译结果   语法分析   句子分析   词汇拓展
               │
               ▼
         Gradio 前端展示
```

### 联调常见 bug 与修复

| Bug | 原因 | 修复 |
|-----|------|------|
| OCR 无输出 | 置信度阈值过高 | 降低 `OCR_CONFIDENCE_THRESHOLD` 到 0.2 |
| 乱码文本 | 图片质量太差 | 开启全部预处理选项，增加光照 |
| LLM 调用超时 | 网络问题或并发过高 | 增加 `LLM_TIMEOUT` 或减少并发任务 |
| 知识库无匹配 | 查询词与库内词汇差异大 | 降低检索时的 k 值，或扩充知识库 |
| FAISS 索引文件找不到 | 未构建索引 | 运行 `python knowledge/build_kb.py` |
| API Key 无效 | 密钥错误或过期 | 检查 LLM_API_KEY 配置，测试连接 |

---

## 七、迭代优化操作

### 7.1 OCR 优化

```python
# config.py 中调整预处理参数
PREPROCESSING = {
    "denoise_strength": 15,     # 增大去噪强度（处理强噪声图片）
    "clahe_clip_limit": 3.0,    # 增强对比度（浅色笔迹）
    "binary_block_size": 21,    # 调整二值化邻域（粗笔迹）
}
```

### 7.2 提示词优化

在 `modules/prompts.py` 的 `PROMPT_VARIANTS` 中维护多个版本，
通过 A/B 对比选择最优提示词写入主模板。

### 7.3 知识库扩容

```bash
# 方式一: 编辑 JSON 文件后重建索引
# 编辑 knowledge/vocabulary.json → 添加新词条
python knowledge/build_kb.py

# 方式二: 通过代码批量导入
python -c "
from modules.knowledge_base import KnowledgeBase
kb = KnowledgeBase()
kb.load_data()
kb.build_index()
kb.batch_import('new_words.json')
"
```

---

## 八、目录结构

```
English OCR/
├── app.py                    # Gradio 入口（启动系统）
├── config.py                 # 全局配置（API Key、参数、路径）
├── requirements.txt          # Python 依赖清单
├── README.md                 # 本文档
├── deploy.md                 # Hugging Face Spaces 部署指南
│
├── modules/                  # 核心功能模块
│   ├── __init__.py
│   ├── preprocessing.py      # 图像预处理（灰度/降噪/二值化/倾斜矫正）
│   ├── ocr_engine.py         # EasyOCR 封装引擎
│   ├── llm_client.py         # 大模型 API 客户端（Qwen/GLM/OpenAI）
│   ├── prompts.py            # 分层提示词模板（5 种任务）
│   └── knowledge_base.py     # FAISS 向量知识库
│
├── utils/                    # 工具模块
│   ├── __init__.py
│   └── text_cleaner.py       # OCR 文本清洗（乱码过滤/混淆修正）
│
└── knowledge/                # 知识库数据
    ├── vocabulary.json       # 考研/四六级高频词汇（~180 条）
    ├── grammar_errors.json   # 常见语法错误模式（~20 条）
    ├── examples.json         # 标准写作例句（~10 条）
    ├── build_kb.py           # 知识库索引构建脚本
    ├── faiss_index.bin       # FAISS 向量索引（自动生成）
    └── index_metadata.json   # 索引元数据（自动生成）
```

---

## 九、技术栈总结

| 层级 | 技术 | 用途 |
|------|------|------|
| 前端 | Gradio 4.x | 可视化 Web 交互界面 |
| OCR | EasyOCR | 手写英文文字识别 |
| 图像处理 | OpenCV + scikit-image | 灰度化/降噪/二值化/倾斜矫正 |
| 知识库 | FAISS + Sentence-Transformers | 向量语义检索 |
| 大模型 | OpenAI 兼容 API | 翻译/纠错/句子分析/词汇拓展 |
| 后端 | Python 3.9+ | 业务逻辑编排 |
| 部署 | Hugging Face Spaces | 免费云端托管 |

---

## 十、作业提交清单

- [x] 源代码（全部 `.py` 文件 + 注释）
- [x] 知识库数据（`.json` 文件）
- [x] 依赖清单（`requirements.txt`）
- [x] 部署文档（`deploy.md`）
- [x] 使用说明（`README.md`）
- [ ] 运行截图（上传图片→OCR→分析结果各截图一张）
- [ ] 公开访问链接（Hugging Face Spaces URL）

---

**Author**: 人工智能导论课程设计
**Date**: 2026-06
**License**: MIT
