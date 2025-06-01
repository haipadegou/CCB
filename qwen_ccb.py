import re
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from pypinyin import pinyin, Style
import threading
import torch

MAX_LENGTH = 70
MIN_LENGTH = 30
PROMPT = '根据用户输入的主题或提示写一篇大约50字的文章。文章的形式是古文，但是可以使用少量现代或英文词汇。文章需要尽可能的好懂并且和主题相关。只输出文章，不要输出其他内容（例如标题或字数），不要问多余的问题。'

SYMBOL = '，。？！：；、 \n'
EOS_SYMBOL = '。？！；\n'


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
    if not (token.isascii() or 0x4E00 <= ord(token[0]) <= 0x9FFF):
        return categories

    # 判断是否为英文单词
    is_english = token.isascii()

    if 'c' in first and (second == 'c' or second == ''):
        categories.add(0 if not is_english else 3)  # cc中文/英文

    if 'c' in first and (second == 'b' or second == ''):
        categories.add(1 if not is_english else 4)  # cb中文/英文

    if 'b' in first and (second == 'c' or second == ''):
        categories.add(2 if not is_english else 5)  # bc中文/英文

    return categories


def classify_vocab(tokenizer):
    """分类整个词汇表"""
    classified = [[], [], [], [], [], [], [], []]  # ccchi cbchi bcchi cceng cbeng bceng symbol eos_symbol

    for token_id in range(tokenizer.vocab_size):
        try:
            token = tokenizer.decode([token_id])
        except:
            continue
        # 跳过特殊token
        if token.startswith("<") and token.endswith(">"):
            continue

        if token[0] in EOS_SYMBOL and not token[1:].isalpha():
            classified[7].append(token_id)

        if token in SYMBOL and len(token) == 1:
            classified[6].append(token_id)
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


def get_incomplete_sentence_ids(token_ids, EOS_SYMBOL):
    # 从后往前查找最后一个句号的位置
    last_eos_pos = -1
    for i in range(len(token_ids) - 1, -1, -1):
        if token_ids[i] in EOS_SYMBOL:
            last_eos_pos = i
            break

    # 如果找到了句号且句号是最后一个元素，则最后一个句子是完整的
    if last_eos_pos != -1 and last_eos_pos == len(token_ids) - 1:
        return []
    # 如果找到了句号但不在末尾，则返回句号之后的所有token
    elif last_eos_pos != -1:
        return token_ids[last_eos_pos + 1:]
    # 如果整个列表中没有句号，则返回整个列表
    else:
        return token_ids.copy()


class CCB_AI(object):
    def __init__(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        self.sent_model = AutoModelForCausalLM.from_pretrained('model', device_map="auto",
                                                               quantization_config=quantization_config)
        self.tokenizer = AutoTokenizer.from_pretrained('model')
        self.valid_model = AutoModelForCausalLM.from_pretrained('qwen-ccb', device_map="auto",
                                                                quantization_config=quantization_config)
        print('使用设备：', self.sent_model.device)

        self.classified = classify_vocab(self.tokenizer)
        self.min_length = 30
        self.start = 0

    def sent_allowed_tokens(self, batch_id, input_ids):
        ans = ' ' + self.tokenizer.decode(input_ids)[self.start:]
        n = count_chinese_and_english(ans)
        valid = self.filter(self.get_sent(input_ids).unsqueeze(0).to(self.sent_model.device), self.classified[n % 3]) + self.classified[n % 3 + 3]
        if ans[-1] not in SYMBOL:
            valid += self.classified[6]
        if n >= self.min_length:
            valid += [self.tokenizer.eos_token_id]
        return valid

    def generate(self, prompt, streamer, min_length=30):
        self.min_length = min_length
        messages = [
            {'role': 'system', 'content': PROMPT},
            {'role': 'user', 'content': prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.sent_model.device)
        self.start = len(text)

        thread = threading.Thread(
            target=self.sent_model.generate,
            kwargs={
                "input_ids": model_inputs["input_ids"],
                "attention_mask": model_inputs["attention_mask"],
                "max_new_tokens": MAX_LENGTH,
                "prefix_allowed_tokens_fn": self.sent_allowed_tokens,
                "temperature": 1.0,
                "top_p": 0.8,
                "use_cache": True,
                "do_sample": True,
                "streamer": streamer,  # 关键：使用 Streamer
                "repetition_penalty": 1.5,
            }
        )
        thread.start()

    def explain(self, prompt, sentence, streamer):
        messages = [
            {"role": "user", "content": f"用“{prompt}”为主题写一段文章，使句子中每个字的拼音首字母交替是C和C和B"},
            {"role": "assistant", "content": sentence},
            {"role": "user", "content": "解释一下这段CCB文章的意思"},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.sent_model.device)

        thread = threading.Thread(
            target=self.sent_model.generate,
            kwargs={
                "input_ids": model_inputs["input_ids"],
                "attention_mask": model_inputs["attention_mask"],
                "max_new_tokens": 200,
                "temperature": 0.5,
                "use_cache": True,
                "do_sample": True,
                "streamer": streamer,  # 关键：使用 Streamer
            }
        )
        thread.start()

    def filter(self, inputs, token_ids, k=0.025):
        if inputs.shape[1] == 0:
            return token_ids
        # 获取模型输出
        with torch.no_grad():
            outputs = self.valid_model(inputs)

        # 获取最后一个位置的logits
        logits = outputs.logits[0, -1, :]

        # 计算所有token的概率
        probs = torch.softmax(logits, dim=-1)

        # 只保留候选token IDs的概率
        candidate_probs = probs[torch.tensor(token_ids, device=self.sent_model.device)]

        # 找到候选token中的最高概率（而不是所有token中的最高概率）
        max_prob_in_candidates = torch.max(candidate_probs)

        # 计算阈值
        threshold = max_prob_in_candidates * k

        # 筛选出概率大于阈值的token
        mask = candidate_probs > threshold
        selected_indices = torch.nonzero(mask).squeeze(-1)

        # 获取对应的token IDs和概率
        selected_tokens = [token_ids[i] for i in selected_indices.tolist()]

        # 按概率从高到低排序
        sorted_indices = torch.argsort(candidate_probs[selected_indices], descending=True)
        selected_tokens = [selected_tokens[i] for i in sorted_indices.tolist()]

        return selected_tokens

    def get_sent(self, token_ids):
        last_eos_pos = -1
        for i in range(len(token_ids) - 1, -1, -1):
            if token_ids[i] in self.classified[7]:
                last_eos_pos = i
                break

        # 如果找到了句号且句号是最后一个元素，则最后一个句子是完整的
        if last_eos_pos != -1 and last_eos_pos == len(token_ids) - 1:
            return torch.tensor([], dtype=torch.int)
        # 如果找到了句号但不在末尾，则返回句号之后的所有token
        elif last_eos_pos != -1:
            return token_ids[last_eos_pos + 1:].clone().detach()
        # 如果整个列表中没有句号，则返回整个列表
        else:
            return token_ids.clone().detach()


if __name__ == '__main__':
    ai = CCB_AI()
    streamer = TextIteratorStreamer(ai.tokenizer, skip_prompt=True, skip_special_tokens=True)
    while True:
        text = input('输入主题：')
        print("生成结果：")
        sentence = ''
        ai.generate(text, streamer, MIN_LENGTH)
        for new_text in streamer:
            print(new_text, end="", flush=True)  # 实时打印
            sentence += new_text
        print()
        ai.explain(text, sentence, streamer)
        print("解释：", end="", flush=True)
        for new_text in streamer:
            print(new_text, end="", flush=True)  # 实时打印
        print()
