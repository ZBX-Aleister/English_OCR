# 🚀 线上部署指南 - Hugging Face Spaces

## 一、部署前准备

### 1.1 注册 Hugging Face 账号
- 访问 https://huggingface.co/join
- 使用邮箱注册（免费）
- 验证邮箱后登录

### 1.2 准备项目文件
确保以下文件在项目根目录：
```
English OCR/
├── app.py                  # Gradio 入口文件（必须命名为 app.py）
├── requirements.txt        # Python 依赖
├── config.py               # 配置文件
├── modules/                # 核心模块
│   ├── preprocessing.py
│   ├── ocr_engine.py
│   ├── llm_client.py
│   ├── prompts.py
│   └── knowledge_base.py
├── utils/
│   └── text_cleaner.py
├── knowledge/              # 知识库数据
│   ├── vocabulary.json
│   ├── grammar_errors.json
│   ├── examples.json
│   └── build_kb.py
└── README.md
```

### 1.3 创建 Git 仓库
```bash
cd "English OCR"

# 初始化 Git
git init

# 创建 .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo ".env" >> .gitignore
echo "knowledge/faiss_index.bin" >> .gitignore
echo "knowledge/index_metadata.json" >> .gitignore

# 提交代码
git add .
git commit -m "Initial commit: English OCR Assistant"
```

## 二、Hugging Face Spaces 部署步骤

### Step 1: 创建 Space
1. 登录 https://huggingface.co
2. 右上角点击头像 → **New Space**
3. 填写 Space 信息：
   - **Space Name**: `english-ocr-assistant`（自定义）
   - **License**: MIT
   - **SDK**: **Gradio**
   - **SDK Version**: 最新版
   - **Hardware**: **CPU basic** (免费) — EasyOCR 在 CPU 上也能正常运行
   - **Visibility**: **Public** (满足公开访问要求)
4. 点击 **Create Space**

### Step 2: 设置 API Key（重要！）
1. 进入 Space 页面
2. 点击 **Settings** 标签
3. 找到 **Repository Secrets** 区域
4. 点击 **Add a secret**：
   - **Name**: `LLM_API_KEY`
   - **Value**: 你的 API Key（通义千问/GLM/OpenAI）
5. 点击 **Save**

### Step 3: 推送代码
```bash
# 添加 Hugging Face 远程仓库
git remote add space https://huggingface.co/spaces/<your-username>/english-ocr-assistant

# 推送代码
git push space main
```

Hugging Face 会自动：
1. 检测到 `app.py` 作为 Gradio 入口
2. 根据 `requirements.txt` 安装依赖
3. 首次启动时下载 EasyOCR 和 Sentence-Transformers 模型
4. 启动应用

### Step 4: 验证部署
1. 在 Space 页面查看 **Building** 日志
2. 等待部署完成（首次约 5-10 分钟，因为要下载模型）
3. 部署完成后，页面会显示 **Running** 状态
4. 你的应用已经在运行！

## 三、公开访问链接

### 链接格式
```
https://huggingface.co/spaces/<your-username>/english-ocr-assistant
```

示例: `https://huggingface.co/spaces/zhangsan/english-ocr-assistant`

### 链接有效期
- **永久有效** — 只要 Space 保持运行状态
- 免费版 Space 在**连续 48 小时无访问**后会进入**休眠 (Sleep)** 模式
- 有人再次访问时自动唤醒（首次访问可能需要等待 2-5 分钟冷启动）
- 休眠不会丢失任何数据或配置

### 保持活跃的方法
- 每月访问一次即可保持活跃
- 如需 7×24 在线，可升级到付费硬件（$0.03~$0.40/小时）

## 四、配置文件修改（部署版）

部署时需要修改 `config.py` 中的几项：

```python
# config.py - 部署版本修改

# 1. 知识库路径（Hugging Face Spaces 中路径不同）
KB_DIR = Path("/tmp/knowledge")  # 使用 /tmp 目录

# 2. 关闭本地 share（Spaces 自己提供公网链接）
GRADIO_SHARE = False

# 3. API Key 通过 Secret 获取（无需修改，已支持环境变量）
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
```

或者创建一个 `deploy_config.py` 来覆盖这些设置。

## 五、常见问题排查

### 问题 1: Build 失败 "No module named 'xxx'"
**解决**: 检查 `requirements.txt` 是否包含所有依赖，特别是 `easyocr` 和 `sentence-transformers`

### 问题 2: 启动后 OCR 不工作
**解决**: 
- EasyOCR 需要首次下载模型（~68MB），可能超时
- 在 `app.py` 中延迟初始化 OCR 引擎
- 确保 Space 的 `Persistent Storage` 已启用（保存下载的模型）

### 问题 3: API Key 无效
**解决**:
- 确认 Secret 名称是 `LLM_API_KEY`（大小写敏感）
- 重新保存 Secret 后需要**重启 Space**（Factory Reboot）
- 查看 Space 日志确认环境变量是否正确加载

### 问题 4: 内存不足
**解决**:
- CPU Basic 有 16GB RAM，足够运行
- 如果 OOM，考虑：
  - 减小输入图片尺寸
  - 禁用部分预处理步骤
  - 升级到更大的硬件

## 六、作业提交信息模板

```
项目名称: 英语手写 OCR 识别翻译助手
技术栈: Python + EasyOCR + FAISS + Gradio + LLM API
公开访问链接: https://huggingface.co/spaces/<username>/english-ocr-assistant
有效期: 长期有效（免费托管）
源代码: https://huggingface.co/spaces/<username>/english-ocr-assistant/tree/main
```
