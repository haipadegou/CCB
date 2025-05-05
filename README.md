# CCB
>[!IMPORTANT]
>
>现已加入Vercel配置文件, 可在Vercel直接部署无需服务器

我将带领人类开发CCB直到100%

本项目与 Bilibili UP 主 **害怕的狗XGGGGGGGGGGG** 的视频[你挖币吗？我挖CC币！](https://www.bilibili.com/video/BV126ZVYKEVu)一同发布，欢迎关注！
# 在线DEMO
<!--作者需要在合并拉去请求时更换链接内容↓-->
[CCB生成器](https://ccb.focalors.ltd)

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
# 请在下方输入生成句子的模型的名称
SENTENCE_MODEL = "deepseek-reasoner"
# 请在下方输入生成解释的模型的名称
EXPLAIN_MODEL = "deepseek-chat"
```
5. 点击部署
6. 等待部署完成
7. 进入项目设置，可添加自己的域名

> **注意**：如果使用Vercel部署，需要存储数据库的功能（查看网友生成的句子）可能会出现问题
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

# 使用本地模型生成
通过网站收集到的训练数据，我们基于Qwen3-0.6B微调了一个模型，专门用于生成CCB句子。

`train.ipynb`中有训练程序。`train_data.json`包含部分训练数据。为了保护个人隐私，文件中只包含同意公开显示的数据。

网站一共收集到了1000条句子，由于缺少训练数据，模型有一点过拟合，可能会出现以下现象：
- 词汇量少
- 一些字会同时出现，例如例如很多观众喜欢输入赤石，所以“粑”和“吃”经常一起出现
- 一些字会频繁地出现，即使和主题无关
- 知识量少，无法理解一些主题
- 偶尔在句子结尾会出现随机词语

如果一次效果不好可以多试几次。
## 1. 安装依赖
```bash
pip install -r -requirements.txt
```
PyTorch安装方法：[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

## 2. 下载基础模型
在当前文件夹使用以下命令下载模型（不需要登录）：
```bash
huggingface-cli download Qwen/Qwen3-0.6B-Base --local-dir qwen3
```

## 3. 运行
运行程序后，通过命令行交互输入主题，并生成符合规则的 CCB 句子。
```bash
python ai_ccb_generator.py
```

## 4. 调整参数
你可以修改程序中的以下常数来调整生成行为：
- `MAX_LENGTH`（长度上限）
- `MIN_LENGTH`（长度下限）
- `CONTROL_OUTPUT` (是否程序化控制输出)
- `TEMPERATURE` (温度)

## 5. 生成解释（可选）
从 [Hugging Face](https://huggingface.co/) 下载任意大语言模型，并将其存放到 `model` 文件夹，不要选择 **预训练阶段的模型**（例如上文要求下载的Qwen模型）。此步骤不做也不会影响程序正常运行。

例如，下载Llama-3.1可以使用以下命令（需要先登录）：
```bash
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir model
```

## 6. 贡献数据
感谢食堂里面开火车、咕噜咕噜、许永豪、阿尔都塞的睡衣、老八、功德台湾、天上飞吧鸡管路、塞琳、紫塔制造机、Bronie、not_beting、not_beting、CCB大蛇、草超棒、和所有匿名用户贡献的训练数据。
