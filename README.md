# CCB
CCB 是一个根据主题生成 CCB 句子的 Python 工具

本项目与 Bilibili UP 主 **害怕的狗XGGGGGGGGGGG** 的视频一同发布，欢迎关注！

此外，本项目还包含一个辅助程序 **CCB 词汇显示工具**，用于显示所有符合 CCB 规则的词汇。
## 技术栈
- Python
- [Transformers](https://huggingface.co/docs/transformers/index)

## 安装
### 1. 安装依赖
```bash
pip install transformers
pip install pypinyin
```
pytorch安装方法：https://pytorch.org/get-started/locally/

### 2. 下载模型
从 [Hugging Face](https://huggingface.co/) 下载任意大模型，并将其存放到 `model` 文件夹。
```bash
huggingface-cli download <模型名称> --local-dir model
```

## 使用方法
### 1. 生成 CCB 句子
运行程序后，通过命令行交互输入主题，并生成符合规则的 CCB 句子。
```bash
python ai_ccb_generator.py
```

### 2. 显示所有 CCB 词汇
运行辅助程序，查看所有符合 CCB 规则的词汇。
```bash
python explorer.py
```

