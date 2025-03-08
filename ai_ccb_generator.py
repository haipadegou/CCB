import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from pypinyin import pinyin, Style

MIN_LENGTH = 10
MAX_LENGTH = 30
SYS_PROMPT = f'''请根据用户提供的主题，按以下优先级生成至少{MIN_LENGTH}个字的中文短句（无需标点）：

第一优先级：主题相关性

句子必须围绕用户给定的主题（如“春天”“学校”等），优先使用与主题直接相关的词汇，句子至少有{MIN_LENGTH}个字。

示例：
✅ 主题“春天” → 花开风吹过（优先主题）
❌ 主题“春天” → 汽车奔跑（偏离主题）

第二优先级：通顺性

句子需自然可读，避免生硬拼接，可接受名词组合（如“晨读书声”）。

示例：
✅ 春风拂面（通顺）
❌ 春菜笔（语义断裂）

第三优先级：CCB结构（尽量满足）

若不影响前两条规则，尽量让每个字的拼音首字母按 C→C→B→C→C→B 循环（可截断）。

结构示例：
✅ 春晨碧草翠波（C-C-B-C-C-B，符合）
✅ 聪才笔创（C-C-B-C，符合）
❌ 校园跑（X-Y-P，首字母错误）'''



class CCB_AI(object):
    def __init__(self):

        model_name = "model"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype='auto',
            device_map="auto"
        )
        self.model = torch.compile(model)

        print('使用设备：', self.model.device)

        self.c_valid_tokens = []
        self.b_valid_tokens = []
        for token_id in range(self.tokenizer.vocab_size):
            word = self.tokenizer.decode([token_id])
            if len(word) == 1 and 'c' in pinyin(word, style=Style.FIRST_LETTER, heteronym=True)[0]:
                self.c_valid_tokens.append(token_id)
            if len(word) == 1 and 'b' in pinyin(word, style=Style.FIRST_LETTER, heteronym=True)[0]:
                self.b_valid_tokens.append(token_id)

        self.index = 0

    def prefix_allowed_tokens_fn(self, batch_id, input_ids):
        self.index += 1
        if self.index > MIN_LENGTH:
            self.c_valid_tokens.append(self.tokenizer.eos_token_id)
            self.b_valid_tokens.append(self.tokenizer.eos_token_id)
        if self.index % 3:
            return self.c_valid_tokens
        return self.b_valid_tokens

    def generate(self, prompt):
        self.index = 0
        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": '主题：' + prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        size = model_inputs['input_ids'].shape[1]

        # 1. 创建 Streamer 用于流式输出
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # 2. 启动一个线程执行 `generate`，避免阻塞
        thread = threading.Thread(
            target=self.model.generate,
            kwargs={
                "input_ids": model_inputs["input_ids"],
                "attention_mask": model_inputs["attention_mask"],
                "max_length": size + MAX_LENGTH,
                "prefix_allowed_tokens_fn": self.prefix_allowed_tokens_fn,
                "temperature": 0.7,
                "use_cache": True,
                "do_sample": True,
                "streamer": streamer  # 关键：使用 Streamer
            }
        )
        thread.start()

        # 3. 逐步输出结果
        print("生成结果：", end="", flush=True)
        for new_text in streamer:
            print(new_text, end="", flush=True)  # 实时打印

        print()  # 换行
        self.c_valid_tokens.pop()
        self.b_valid_tokens.pop()


if __name__ == '__main__':
    ai = CCB_AI()
    while True:
        text = input('输入主题：')
        print('正在生成（可能需要几分钟）')
        ai.generate(text)
