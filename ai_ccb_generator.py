import os
import re
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from pypinyin import pinyin, Style
import threading
import torch

MAX_LENGTH = 30
MIN_LENGTH = 4
CONTROL_OUTPUT = True
TEMPERATURE = 0.7

SYMBOL = ',.?!:;，。？！：；、 \n'


def get_pinyin_initials(input_str):
    # 英文单词：返回首字母 + 空字符串
    if input_str.isascii():
        return input_str[0].lower(), ""
    # 处理中文
    if len(input_str) == 1:
        # 单字：获取所有可能的拼音首字母
        all_pinyin = pinyin(input_str, style=Style.NORMAL, heteronym=True)  # 如 [['zhong'], ['chong']]
        initials = {py[0][0].lower() for py in all_pinyin}  # 去重并转为小写
        return "".join(sorted(initials)), ""  # 按字母顺序排列
    else:
        # 两字及以上：默认词语读音唯一，直接取首字母
        pinyin_initials = pinyin(input_str, style=Style.FIRST_LETTER)
        if len(pinyin_initials) < 2:
            return "", ""
        return pinyin_initials[0][0], pinyin_initials[1][0]


def classify_token(token):
    """按扩展规则分类token"""
    first, second = get_pinyin_initials(token)
    categories = set()
    if not (token.isascii() or 0x4E00 <= ord(token[0]) <= 0x9FFF or token in SYMBOL):
        return categories

    if 'c' in first and (second == 'c' or second == '') or token in SYMBOL:
        categories.add(0)  # cc

    if 'c' in first and (second == 'b' or second == '') or token in SYMBOL:
        categories.add(1)  # cb

    if 'b' in first and (second == 'c' or second == '') or token in SYMBOL:
        categories.add(2)  # bc

    return categories


def classify_vocab(tokenizer):
    """分类整个词汇表"""
    classified = [[], [], []]

    for token_id in range(tokenizer.vocab_size):
        try:
            token = tokenizer.decode([token_id])
        except:
            continue
        # 跳过特殊token
        if token.startswith("<") and token.endswith(">"):
            continue

        categories = classify_token(token)
        for cat in categories:
            classified[cat].append(token_id)

    return classified


def count_chinese_and_english(text):
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    chinese_count = len(chinese_chars)

    english_words = re.findall(r"[a-zA-Z']+", text)
    english_count = len(english_words)

    return chinese_count + english_count


class CCB_AI(object):
    def __init__(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        base_model = AutoModelForCausalLM.from_pretrained('qwen3', device_map="auto",
                                                           quantization_config=quantization_config)
        self.sent_model = PeftModel.from_pretrained(base_model, "qwen-ccb")
        self.tokenizer = AutoTokenizer.from_pretrained('qwen-ccb')
        self.exp_model = None
        self.exp_tokenizer = None
        if os.path.isdir('model'):
            self.exp_model = AutoModelForCausalLM.from_pretrained('model', device_map="auto",
                                                                  quantization_config=quantization_config)
            self.exp_tokenizer = AutoTokenizer.from_pretrained('model')

        print('使用设备：', self.sent_model.device)

        self.classified = classify_vocab(self.tokenizer)
        self.min_length = 4

    def sent_allowed_tokens(self, batch_id, input_ids):
        ans = self.tokenizer.decode(input_ids)
        ans = ans.split('### ASSISTANT:')[1]
        n = count_chinese_and_english(ans)
        if n >= self.min_length:
            return self.classified[n % 3] + [self.tokenizer.eos_token_id]
        else:
            return self.classified[n % 3]

    def generate(self, prompt, min_length=4, temp=0.7):
        self.min_length = min_length
        if CONTROL_OUTPUT:
            fun = self.sent_allowed_tokens
        else:
            fun = None
        model_inputs = self.tokenizer(f'### USER: {prompt}\n### ASSISTANT:', return_tensors="pt").to(
            self.sent_model.device)
        out = self.sent_model.generate(**model_inputs, max_new_tokens=MAX_LENGTH, temperature=temp,
                                       prefix_allowed_tokens_fn=fun, use_cache=True, do_sample=True)
        out = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return out

    def explain(self, prompt, sentence, streamer):
        messages = [
            {"role": "user", "content": f"用“{prompt}”为主题生成一个句子，使句子中每个字的拼音首字母交替是C和C和B"},
            {"role": "assistant", "content": sentence},
            {"role": "user", "content": "这个句子是什么意思？"},
        ]
        print(messages)
        text = self.exp_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.exp_tokenizer([text], return_tensors="pt").to(self.exp_model.device)
        size = model_inputs['input_ids'].shape[1]

        thread = threading.Thread(
            target=self.exp_model.generate,
            kwargs={
                "input_ids": model_inputs["input_ids"],
                "attention_mask": model_inputs["attention_mask"],
                "max_length": size + MAX_LENGTH * 2,
                "temperature": 0.5,
                "use_cache": True,
                "do_sample": True,
                "streamer": streamer  # 关键：使用 Streamer
            }
        )
        thread.start()


if __name__ == '__main__':
    ai = CCB_AI()
    streamer = TextIteratorStreamer(ai.tokenizer, skip_prompt=True, skip_special_tokens=True)
    while True:
        text = input('输入主题：')
        print("生成结果：")
        sentence = ai.generate(text, MIN_LENGTH, TEMPERATURE)
        print(sentence)
        if not (ai.exp_model is None):
            ai.explain(text, sentence, streamer)
            print("解释：", end="", flush=True)
            for new_text in streamer:
                print(new_text, end="", flush=True)  # 实时打印
            print()
