from flask import Flask, request, render_template, jsonify
from openai import OpenAI
from pypinyin import pinyin, Style
from threading import Semaphore, Thread
import uuid
import os
import json
from dotenv import load_dotenv  # 新增导入

load_dotenv()  # 新增环境变量加载

app = Flask(__name__)
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # 修改为环境变量读取方式
    base_url="https://api.chatanywhere.tech/v1"
)

# 限制最大并发任务数
MAX_CONCURRENT_REQUESTS = 1
semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)

# 存储任务状态
tasks = {}  # 结构: {task_id: {"status": "pending", "sentences": [], "explanation": None, "error": None}}

def read_char(fn):
    with open(fn, encoding='utf-8') as f:
        data = f.read().split('\n')
        f.close()
        i = 0
        while i < len(data):
            if len(data[i]) == 0:
                data.pop(i)
            else:
                i += 1
    return data


# 模拟文件系统的路径
BASE_DIR = "ccblist"
INDEX_FILE = os.path.join(BASE_DIR, "index.json")
all_c, all_b = [], []
with open(INDEX_FILE, "r", encoding="utf-8") as f:
    index_data = f.read()
langs = json.loads(index_data)
for key in langs.keys():
    if key != '所有语言':
        lang_path = os.path.join(BASE_DIR, langs[key])
        c_file = os.path.join(lang_path, "c.txt")
        b_file = os.path.join(lang_path, "b.txt")
        all_c += read_char(c_file)
        all_b += read_char(b_file)


# ---------句子生成---------

def chat(model, messages, temperature=1):
    response = client.chat.completions.create(model=model, messages=messages, stream=False, temperature=temperature)
    return response.choices[0].message.content


def validate(txt, min_length):
    """ 验证生成的句子是否符合规则 """
    txt = txt.lower()
    p = pinyin(txt, style=Style.FIRST_LETTER)
    j = 0
    res = True
    for i in range(len(p)):
        if txt[i] in "，。！":
            continue
        j += 1
        if j % 3:
            if p[i][0] != "c":
                res = False
                break
        elif p[i][0] != "b":
            res = False
            break
    if len(p) < min_length:
        return "句子长度太短", txt[:i]
    elif res:
        return "ok", txt
    elif i >= min_length:
        return "ok", txt[:i]
    else:
        return f"第{i + 1}个字“{txt[i]}”的首字母出错", txt[:i]


def generate(theme, min_length, temperature, messages=None):
    """ 生成句子 """
    prompt = f'''请生成一句与“{theme}”相关的中文短句。
    每个字的拼音首字母必须严格按照这个规则：首字母交替为C和C和B，例如C-C-B-C-C-B-C-C-B。
    句子长度至少是{min_length}，不需要是3的倍数，结尾可以截断，例如结尾可以是B-C或B-C-C。
    句子需要和主题相关，通顺，易于理解。主要使用中文，可以使用俚语。
    输出方式：每个句子一行，不要输出其他的。
    输出之前仔细检查句子首字母是否符合要求，确保句子一定符合规则。'''

    if messages is None:
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ]
    content = chat("deepseek-reasoner", messages, temperature)
    messages.append({"role": "assistant", "content": content})

    return messages


def explain(messages):
    """ 生成解释 """
    messages.append({"role": "user", "content": "把这些你输出的句子翻译成正常的语言"})
    return chat("deepseek-chat", messages, 1)


def run_task(task_id, theme, length_min, temp):
    """ 任务执行逻辑 """
    if not semaphore.acquire(blocking=False):
        tasks[task_id]["error"] = "服务器繁忙，请稍后再试"
        tasks[task_id]["status"] = "failed"
        return

    try:
        messages = None
        best_sen = ''
        for _ in range(5):  # 最多尝试 5 次
            messages = generate(theme, length_min, temp, messages)
            sentence = messages[-1]["content"]
            res, data = validate(sentence, length_min)
            if len(data) > len(best_sen):
                best_sen = data
            if res == 'ok':
                tasks[task_id]["sentences"].append(({'sentence': data, 'valid': True}))  # 存储句子
                explanation = explain(messages)
                tasks[task_id]["explanation"] = explanation
                tasks[task_id]["status"] = "completed"
                break
            else:
                tasks[task_id]["sentences"].append(({'sentence': sentence, 'valid': False}))  # 存储句子
                messages.append({"role": "user", "content": "生成的句子有问题，" + res})
        else:
            tasks[task_id]["error"] = "无法生成符合规则的句子，请尝试更简单的主题，或者降低句子长度下限"
            tasks[task_id]["status"] = "failed"

    except Exception as e:
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["status"] = "failed"
    finally:
        semaphore.release()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit_task():
    """ 提交新任务 """
    theme = request.json.get("theme")
    length_min = int(request.json.get("length_min"))
    temp = float(request.json.get("temperature"))

    if not theme:
        return jsonify({"error": "请输入主题"}), 400

    task_id = str(uuid.uuid4())  # 生成唯一任务 ID
    tasks[task_id] = {"status": "pending", "sentences": [], "explanation": None, "error": None}
    # 异步执行任务
    thread = Thread(target=run_task, args=(task_id, theme, length_min, temp))
    thread.start()

    return jsonify({"task_id": task_id}), 202


@app.route("/status/<task_id>", methods=["GET"])
def get_task_status(task_id):
    """ 获取任务状态 """
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "任务不存在", "status": "failed"}), 404

    if task["status"] in ["completed", "failed"]:
        result = task.copy()  # 复制数据
        del tasks[task_id]  # 删除任务释放内存
        return jsonify(result)

    return jsonify(task)


# ------------------CCB词查看--------------


@app.route("/explorer", methods=["GET"])
def explorer():
    return render_template('explorer.html')


@app.route("/explorer/index", methods=["GET"])
def get_index():
    """返回 index.json 内容"""
    return index_data


@app.route("/explorer/language", methods=["GET"])
def get_language_files():
    """获取指定语言的 c.txt 和 b.txt 文件内容"""
    lang = request.args.get("lang")

    if lang == '所有语言':
        return jsonify({
            "c": all_c,
            "b": all_b
        })
    if lang not in langs:
        return jsonify({"error": "Language not found"}), 404

    lang_path = os.path.join(BASE_DIR, langs[lang])
    c_file = os.path.join(lang_path, "c.txt")
    b_file = os.path.join(lang_path, "b.txt")

    return jsonify({
        "c": read_char(c_file),
        "b": read_char(b_file)
    })


if __name__ == "__main__":
    app.run(debug=True)
