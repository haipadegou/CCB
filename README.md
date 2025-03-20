# CCB
我将带领人类开发CCB直到100%

本项目与 Bilibili UP 主 **害怕的狗XGGGGGGGGGGG** 的视频一同发布，欢迎关注！

## 技术栈
- Python
- Flask

## 安装
### 1. 安装依赖
```bash
pip install flask
pip install openai
pip install pypinyin
```

### 2. 填写API key
在`app.py`的第10行填入DeepSeek API key

## 使用方法
运行程序，然后进入网站[http://127.0.0.1:5000](http://127.0.0.1:5000/)
```bash
python app.py
```

## 使用本地模型生成
如果有高性能设备，可以选择运行本地模型而不是使用DeepSeek API。

因为transformers可以修改模型输出的概率分布，所以可以程序化的控制输出，不需要通过深度思考来检查句子。这可以减少句子生成的时间。
### 1. 安装依赖
```bash
pip install transformers
pip install pypinyin
```
PyTorch安装方法：[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### 2. 下载模型
从 [Hugging Face](https://huggingface.co/) 下载任意大语言模型，并将其存放到 `model` 文件夹，不要选择 **推理专用模型（reasoning models）**。

在当前文件夹使用以下命令下载模型：
```bash
huggingface-cli download <模型名称> --local-dir model
```

### 3. 运行
运行程序后，通过命令行交互输入主题，并生成符合规则的 CCB 句子。
```bash
python ai_ccb_generator.py
```

### 4. 调整参数
你可以修改程序中的以下常数来调整生成行为：
- `SYS_PROMPT`（系统提示）
- `MIN_LENGTH`（长度下限）
- `MAX_LENGTH`（长度上限）
