from datasets import load_dataset
import re
import tokenize
from io import StringIO
import torch
from torch.utils.data import Dataset, DataLoader

class CodeTokenizer:
    def __init__(self):
        self.special_tokens = {"<|code|>": 0, "<|end|>": 1, "<|indent|>": 2, "<|newline|>": 3, "<|unk|>": 4}
        self.str_to_int = {}
        self.int_to_str = {}
        
    def build_vocab(self, codes):
        tokens = set()
        for code in codes:
            try:
                code_io = StringIO(code)
                for tok in tokenize.generate_tokens(code_io.readline):
                    if tok.string.strip():
                        tokens.add(tok.string)
            except:
                tokens.update(re.findall(r'\w+|[^\w\s]', code))
        
        vocab = list(self.special_tokens.keys()) + sorted(tokens)
        self.str_to_int = {t: i for i, t in enumerate(vocab)}
        self.int_to_str = {i: t for t, i in self.str_to_int.items()}
    
    def encode(self, code):
        tokens = ["<|code|>"]
        lines = code.split('\n')
        prev_indent = 0
        
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                if indent > prev_indent:
                    tokens.extend(["<|indent|>"] * ((indent - prev_indent) // 4))
                
                try:
                    line_io = StringIO(line.strip())
                    for tok in tokenize.generate_tokens(line_io.readline):
                        if tok.string.strip():
                            tokens.append(tok.string)
                except:
                    tokens.extend(re.findall(r'\w+|[^\w\s]', line.strip()))
                
                prev_indent = indent
            tokens.append("<|newline|>")
        
        tokens.append("<|end|>")
        return [self.str_to_int.get(t, self.str_to_int["<|unk|>"]) for t in tokens]

# Load and process dataset
ds = load_dataset("flytech/python-codes-25k")
codes = [ds['train'][i]['code'] for i in range(1000)]

tokenizer = CodeTokenizer()
tokenizer.build_vocab(codes)

# Test
test_code = ds['train'][0]['code']
ids = tokenizer.encode(test_code)

print(f"Dataset size: {len(ds['train'])}")
print(f"Vocab size: {len(tokenizer.str_to_int)}")
print(f"Sample tokenized to {len(ids)} tokens")
print("First tokens:", [tokenizer.int_to_str[i] for i in ids[:10]])
