# CCB
>[!IMPORTANT]
>
>现已加入Vercel配置文件, 可在Vercel直接部署无需服务器

我将带领人类开发CCB直到100%

本项目与 Bilibili UP 主 **害怕的狗XGGGGGGGGGGG** 的视频一同发布，欢迎关注！
# 在线DEMO
<!--作者需要在合并拉去请求时更换链接内容↓-->
[CCB生成器](https://ccb.focalors.ltd)
[体验](http://47.99.50.149/)

## 技术栈
- Python
- Flask

# Vercel部署教程
1. 注册Vercel账号和Github账号
> Vercel注册页面:[点击跳转](https://vercel.com/signup)
> **注意**: 千万不要使用国内邮箱注册, 否则需要开工单
2. 在Github上fork本项目
3. 在Vercel上新建项目, 或者直接[点击跳转](https://vercel.com/new)
4. 选择Github项目
5. 选择fork的项目
6. 在项目设置中，选择环境变量, 添加以下环境变量
> 你也可以直接在仓库页面的`.env.example`文件中填写, 随后将其更名为`.env`(不推荐)
```
# 请在下方输入您的apikey
OPENAI_API_KEY = ""
# 请在下方输入API的url
OPENAI_API_URL = "https://api.deepseek.com"
```
5. 点击部署
6. 等待部署完成
7. 进入项目设置，可添加自己的域名
# 本地部署教程
## 克隆项目
```bash
git clone https://github.com/haipadegou/CCB.git
```

## 安装
### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 填写API url 和 key
在`.env.example`的第填入DeepSeek API url和key, 同时将文件名更换为`.env`

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
pip install -r -requirements.txt
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
