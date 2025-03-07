import tkinter as tk
from tkinter import ttk
import math
import json


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


class VirtualFileSystemApp:
    def __init__(self, root, init_lang):
        self.root = root
        with open('ccblist\\index.json') as f:
            self.langs = json.load(f)
            f.close()

        self.lang = init_lang
        self.read()

        # 存储已经加载的文件夹地址，避免重复加载
        self.loaded_folders = set()

        # 顶部选项栏
        self.option_frame = tk.Frame(self.root)
        self.option_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(self.option_frame, text="选择语言:").pack(side=tk.LEFT, padx=5)
        self.file_selector = ttk.Combobox(self.option_frame, values=list(self.langs.keys()), state="readonly")
        self.file_selector.pack(side=tk.LEFT)
        self.file_selector.current(0)
        self.file_selector.bind("<<ComboboxSelected>>", self.on_file_num_change)

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 创建左侧的文件树 (Treeview)
        self.tree_frame = tk.Frame(self.main_frame, width=1000)
        self.tree_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.tree = ttk.Treeview(self.tree_frame)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<Double-1>", self.on_item_double_click)  # 绑定双击事件
        self.tree.bind("<<TreeviewOpen>>", self.on_treeview_open)  # 绑定展开事件

        # 创建右侧的文本区域，用于显示文件内容
        self.content_frame = tk.Frame(self.main_frame)
        self.content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)

        # 文件名显示区域
        self.file_label = tk.Label(self.content_frame, text="文件内容", font=("Arial", 14, "bold"))
        self.file_label.pack(fill=tk.X, padx=10, pady=5)

        self.text = tk.Text(self.content_frame, wrap=tk.WORD)
        self.text.pack(fill=tk.BOTH, expand=True)
        self.setup_ui()

    def read(self):
        self.c = read_char(f'ccblist\\{self.langs[self.lang]}\\c.txt')
        self.b = read_char(f'ccblist\\{self.langs[self.lang]}\\b.txt')
        self.file_num = len(self.c) ** 2 * len(self.b)
        print(self.file_num)
        self.max_depth = int(math.log10(self.file_num))
        self.root_num = math.ceil(self.file_num / 10 ** self.max_depth)

    def setup_ui(self):
        self.tree.delete(*self.tree.get_children())  # 清空树
        self.loaded_folders.clear()  # 清空加载记录

        self.tree.heading("#0", text=f"{self.lang}的总ccb词数量：{self.file_num}", anchor=tk.W)

        for i in range(self.root_num):
            root_addr = str(i)  # 根文件夹地址
            root_item_id = f"root_{root_addr}"  # 根文件夹的唯一ID
            self.tree.insert("", "end", text=f"文件夹 {root_addr}", open=False, iid=root_item_id)
            self.loaded_folders.add(root_addr)
            self.populate_tree(root_addr, root_item_id)  # 展开根文件夹

    def populate_tree(self, addr, parent, load=True):
        # 获取该地址文件夹的内容，打开文件夹
        items_count = self.open_folder(addr)
        depth = len(addr)

        for i in range(items_count):
            item_addr = f"{addr}{i}"
            item_id = f"folder_{item_addr}"  # 生成唯一的节点ID

            if depth < self.max_depth - 1:
                # 文件夹
                if item_addr not in self.loaded_folders:
                    folder_item = self.tree.insert(parent, "end", text=f"文件夹 {item_addr}", open=False, iid=item_id)
                    self.loaded_folders.add(item_addr)
                if load:
                    self.populate_tree(item_addr, item_id, False)
            else:
                # 文件
                if item_addr not in self.loaded_folders:
                    self.tree.insert(parent, "end", text=f"文件 {item_addr}", open=False, iid=item_id)
                    self.loaded_folders.add(item_addr)

    def on_item_double_click(self, event):
        # 获取双击的树项
        item_id = self.tree.selection()[0]
        item_text = self.tree.item(item_id, "text")

        if not item_text.startswith("文件夹"):
            # 提取文件地址
            file_addr = self.tree.item(item_id, "text").split(" ")[1]
            file_content = self.open_file(int(file_addr))
            # 更新文件名显示
            self.file_label.config(text=item_text)

            # 清空文本区域，并插入文件内容
            self.text.delete(1.0, tk.END)
            self.text.insert(tk.END, file_content)

    def on_treeview_open(self, event):
        # 获取双击的树项
        item_id = self.tree.selection()[0]
        item_text = self.tree.item(item_id, "text")

        folder_addr = item_text.split(" ")[1]

        self.populate_tree(folder_addr, item_id)  # 加载子文件夹

    def on_file_num_change(self, event):
        self.lang = self.file_selector.get()
        self.read()
        self.setup_ui()

    def open_file(self, addr):
        cl, bl = len(self.c), len(self.b)
        product = cl * bl
        ans = ''
        for i in range(addr * 10, addr * 10 + 10):
            x = i // product
            if x >= cl:
                break
            remaining = i % product  # 或者是 n - x * product，结果相同
            y = remaining // bl
            z = remaining % bl
            ans += self.c[x] + ' ' + self.c[y] + ' ' + self.b[z] + '\n'
        return ans

    def open_folder(self, addr):
        depth = self.max_depth - len(addr) + 1
        addr = int(addr)
        if (addr + 1) * 10 ** depth > self.file_num:
            return math.ceil((self.file_num - addr * 10 ** depth) / (10 ** (depth - 1)))
        else:
            return 10


if __name__ == "__main__":
    root = tk.Tk()
    root.title("虚拟CCB文件管理系统")
    app = VirtualFileSystemApp(root, '中文')
    root.mainloop()
