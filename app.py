from flask import Flask, request, render_template, jsonify, Response
from openai import OpenAI
from pypinyin import pinyin, Style
from threading import Semaphore, Thread
import time
import uuid
import sqlite3
import os
import json
from dotenv import load_dotenv  # 新增导入

load_dotenv()  # 新增环境变量加载

app = Flask(__name__)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_API_URL"))

# 限制最大并发任务数
MAX_CONCURRENT_REQUESTS = 1
semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)
last_request = {}

# 存储任务状态
tasks = {}  # 结构: {task_id: {"status": "pending", "sentences": [], "explanation": None, "error": None}}

PROMPT = '''请生成一句与“{theme}”相关的中文短句。
如果你无法理解主题（主题可能是随机无意义字符）就直接输出“无法理解主题”，不要尝试过度解读、把主题看作猜谜或密码、或强行生成句子。
如果你可以理解主题，就按照一下规则生成一句短句：
每个字的拼音首字母必须严格按照这个规则：首字母交替为C和C和B，例如C-C-B-C-C-B-C-C-B。
句子长度至少是{min_length}个字，不需要是3的倍数，结尾可以截断，例如结尾可以是B-C或B-C-C。
句子需要和主题相关，通顺，易于理解。主要使用中文，可以使用俚语或英语。
输出方式：每个句子一行，不要输出其他的；
输出之前仔细检查句子首字母是否符合要求，确保句子一定符合规则。'''

MIN_LENGTH = 10
TEMPERATURE = 1.5
HIS_LEN = 20

SYMBOL = ',.?!;，。？！；'


def read_char(fn):
    with open(fn, encoding='utf-8') as f:  # 已明确指定UTF-8
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
    p_word = []
    word_i = []
    l = 0
    for i in range(len(p)):
        line = p[i][0]
        line += ' '
        word = ''
        for j in range(len(line)):
            if line[j] == ' ':
                if word not in SYMBOL and word != '':
                    p_word.append(word)
                    word_i.append(l + j)
                word = ''
            else:
                word += line[j]
        l += len(p[i][0])

    for i in range(len(p_word)):
        if (i + 1) % 3:
            if p_word[i][0] != "c":
                break
        elif p_word[i][0] != "b":
            break
        ans_txt = txt[:word_i[i]]
    else:
        i += 1
    if len(p_word) < min_length:
        return "句子长度太短", ''
    elif i >= min_length:
        return "ok", ans_txt
    else:
        return f"第{i + 1}个字的首字母出错", ''


def generate(theme, min_length, temperature, messages=None):
    """ 生成句子 """
    prompt = PROMPT.format(theme=theme, min_length=min_length)
    if messages is None:
        messages = [
            {"role": "user", "content": prompt}
        ]
    content = chat(os.environ.get("SENTENCE_MODEL"), messages, temperature)
    messages.append({"role": "assistant", "content": content})

    return messages


def explain(messages):
    """ 生成解释 """
    messages.append({"role": "user", "content": "把这些你输出的句子翻译成正常的语言"})
    return chat(os.environ.get("EXPLAIN_MODEL"), messages, 1)


def save(theme, sentence, explanation, cursor, conn, public):
    global num
    if len(sentence) >= 4:
        max_id = cursor.execute('SELECT MAX(id) FROM sentences').fetchall()[0][0] + 1
        cursor.execute(
            "INSERT INTO sentences (id, time, theme, content, explain, public) VALUES (?, ?, ?, ?, ?, ?)",
            (max_id, time.time(), theme, sentence, explanation, public))
        conn.commit()


def run_task(task_id, theme, length_min, temp, public):
    """ 任务执行逻辑 """
    if len(theme) > 20:
        tasks[task_id]["error"] = "过于复杂或细节的主题不适合生成句子，请使用长度不超过20个字的主题"
        tasks[task_id]["status"] = "failed"
        return
    if not semaphore.acquire(blocking=False):
        tasks[task_id]["error"] = "服务器繁忙，请稍后再试"
        tasks[task_id]["status"] = "failed"
        return

    conn = sqlite3.connect('sentences.db')
    cursor = conn.cursor()

    try:
        messages = None
        best_sen = ''
        for _ in range(5):  # 最多尝试 5 次
            messages = generate(theme, length_min, temp, messages)
            sentence = messages[-1]["content"]
            if '无法理解主题' in sentence:
                tasks[task_id]["error"] = "无法理解主题，让我们说中文"
                tasks[task_id]["status"] = "failed"
                break
            res, data = validate(sentence, length_min)
            if len(data) > len(best_sen):
                best_sen = data
            if res == 'ok':
                messages[-1]["content"] = data
                tasks[task_id]["sentences"].append(({'sentence': data, 'valid': True}))  # 存储句子
                explanation = explain(messages)
                save(theme, data, explanation, cursor, conn, public)
                tasks[task_id]["explanation"] = explanation
                tasks[task_id]["status"] = "completed"
                break
            else:
                tasks[task_id]["sentences"].append(({'sentence': sentence, 'valid': False}))  # 存储句子
                messages.append({"role": "user", "content": "生成的句子有问题，" + res})
        else:
            tasks[task_id]["error"] = "无法生成符合规则的句子，请尝试更简单的主题，或者降低句子长度下限"
            tasks[task_id]["status"] = "failed"
            save(theme, best_sen, '---', cursor, conn, public)

    except Exception as e:
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["status"] = "failed"
    finally:
        semaphore.release()
        conn.close()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit_task():
    """ 提交新任务 """
    theme = request.json.get("theme")
    length_min = int(request.json.get("length_min", MIN_LENGTH))
    temp = float(request.json.get("temperature", TEMPERATURE))
    public = request.json.get("public")

    if not theme:
        return jsonify({"error": "请输入主题"}), 400

    ip = request.remote_addr
    if ip in last_request:
        if time.time() - last_request[ip] < 5:
            return jsonify({"error": "服务器繁忙，请稍后再试"}), 400

    last_request[ip] = time.time()

    task_id = str(uuid.uuid4())  # 生成唯一任务 ID
    tasks[task_id] = {"status": "pending", "sentences": [], "explanation": None, "error": None}
    # 异步执行任务
    thread = Thread(target=run_task, args=(task_id, theme, length_min, temp, public))
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


# -----------历史记录----------------

@app.route('/history')
def history_web():
    return render_template('history.html')


@app.route('/history_data', methods=['POST'])
def history_data():
    conn = sqlite3.connect('sentences.db')
    cursor = conn.cursor()
    page = request.json.get('page')
    data = cursor.execute(
        f'SELECT * FROM sentences WHERE public=1 ORDER BY time DESC LIMIT ? OFFSET ?',
        (HIS_LEN, page * HIS_LEN)).fetchall()
    num = cursor.execute(
        f'SELECT COUNT(*) FROM sentences').fetchall()[0][0]
    for i in range(len(data)):
        t = cursor.execute(
            f'SELECT COUNT(*) FROM comments WHERE sentence_id=?', (data[i][0],)).fetchall()[0][0]
        data[i] += (t,)
    conn.close()
    return jsonify({'num': num, 'sentence': data})


@app.route('/get_comments', methods=['POST'])
def get_comments():
    conn = sqlite3.connect('sentences.db')
    cursor = conn.cursor()
    sent_id = request.json.get('sentence_id')
    data = cursor.execute(f'SELECT * FROM comments WHERE sentence_id=?', (sent_id,)).fetchall()
    conn.close()
    return jsonify(data)


@app.route('/post_comment', methods=['POST'])
def post_comment():
    conn = sqlite3.connect('sentences.db')
    cursor = conn.cursor()
    sent_id = int(request.json.get('sentence_id'))
    txt = request.json.get('text')
    com_num = cursor.execute(f'SELECT MAX(comment_id) FROM comments').fetchall()[0][0] + 1
    cursor.execute(f'INSERT INTO comments (comment_id, sentence_id, time, content) VALUES (?, ?, ?, ?)',
                   (com_num, sent_id, time.time(), txt))
    conn.commit()
    conn.close()
    return 'ok'


# ------------------CCB词查看--------------


@app.route("/explorer", methods=["GET"])
def explorer():
    return render_template('explorer.html')


@app.route("/explorer/index", methods=["GET"])
def get_index():
    """返回 index.json 内容"""
    return Response(index_data, mimetype='application/json; charset=utf-8')


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

    return jsonify({  # Flask默认使用application/json; charset=utf-8
        "c": read_char(c_file),
        "b": read_char(b_file)
    }), 200, {'Content-Type': 'application/json; charset=utf-8'}


# ------------问卷-----------
@app.route("/query")
def query():
    return render_template('query.html')


@app.route("/query_submit", methods=["POST"])
def query_submit():
    sentence = request.json.get('content')
    name = request.json.get('name')
    res, data = validate(sentence, 4)
    if res == 'ok':
        if name == '':
            explain = '这是人工生成的'
        else:
            explain = f'这是由{name}人工生成的'
        conn = sqlite3.connect('sentences.db')
        cursor = conn.cursor()
        save(request.json.get('topic'), data, explain, cursor, conn, True)
        conn.close()
        return jsonify({'ok': True})
    else:
        return jsonify({'ok': False, 'message': res})


if __name__ == "__main__":
    app.run()
