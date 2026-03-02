import re
import ast
import tokenize
from io import StringIO

class PythonCodeTokenizer:
    def __init__(self):
        # Special tokens for code structure
        self.special_tokens = {
            "<|startofcode|>": 0,
            "<|endofcode|>": 1,
            "<|indent|>": 2,
            "<|dedent|>": 3,
            "<|newline|>": 4,
            "<|unk|>": 5
        }
        self.vocab = {}
        self.str_to_int = {}
        self.int_to_str = {}
        
    def build_vocab(self, code_samples):
        """Build vocabulary from Python code samples"""
        all_tokens = set()
        
        for code in code_samples:
            tokens = self._tokenize_code(code)
            all_tokens.update(tokens)
        
        # Create vocabulary
        vocab_list = list(self.special_tokens.keys()) + sorted(all_tokens)
        self.str_to_int = {token: i for i, token in enumerate(vocab_list)}
        self.int_to_str = {i: token for token, i in self.str_to_int.items()}
        
    def _tokenize_code(self, code):
        """Tokenize Python code preserving structure"""
        tokens = []
        try:
            # Use Python's tokenize module
            code_io = StringIO(code)
            for tok in tokenize.generate_tokens(code_io.readline):
                if tok.type != tokenize.ENCODING:
                    token_str = tok.string
                    if token_str.strip():  # Skip empty tokens
                        tokens.append(token_str)
        except:
            # Fallback to simple splitting
            tokens = re.findall(r'\w+|[^\w\s]', code)
        
        return tokens
    
    def encode(self, code):
        """Encode Python code to token IDs"""
        tokens = ["<|startofcode|>"]
        
        # Handle indentation
        lines = code.split('\n')
        prev_indent = 0
        
        for line in lines:
            if line.strip():
                # Calculate indentation
                current_indent = len(line) - len(line.lstrip())
                
                # Add indent/dedent tokens
                if current_indent > prev_indent:
                    tokens.extend(["<|indent|>"] * ((current_indent - prev_indent) // 4))
                elif current_indent < prev_indent:
                    tokens.extend(["<|dedent|>"] * ((prev_indent - current_indent) // 4))
                
                # Tokenize the line
                line_tokens = self._tokenize_code(line.strip())
                tokens.extend(line_tokens)
                prev_indent = current_indent
            
            tokens.append("<|newline|>")
        
        tokens.append("<|endofcode|>")
        
        # Convert to IDs
        ids = []
        for token in tokens:
            if token in self.str_to_int:
                ids.append(self.str_to_int[token])
            else:
                ids.append(self.str_to_int["<|unk|>"])
        
        return ids
    
    def decode(self, ids):
        """Decode token IDs back to Python code"""
        tokens = [self.int_to_str.get(id, "<|unk|>") for id in ids]
        
        code_lines = []
        current_line = []
        indent_level = 0
        
        for token in tokens:
            if token == "<|startofcode|>":
                continue
            elif token == "<|endofcode|>":
                break
            elif token == "<|indent|>":
                indent_level += 1
            elif token == "<|dedent|>":
                indent_level = max(0, indent_level - 1)
            elif token == "<|newline|>":
                if current_line:
                    line = "    " * indent_level + " ".join(current_line)
                    code_lines.append(line)
                    current_line = []
                else:
                    code_lines.append("")
            else:
                current_line.append(token)
        
        return "\n".join(code_lines)

# Usage with sample Python code
if __name__ == "__main__":
    sample_codes = [
        "def hello():\n    print('Hello')",
        "for i in range(10):\n    x = i * 2"
    ]
    
    tokenizer = PythonCodeTokenizer()
    tokenizer.build_vocab(sample_codes)
    
    test_code = sample_codes[0]
    ids = tokenizer.encode(test_code)
    decoded = tokenizer.decode(ids)
    
    print(f"Vocab size: {len(tokenizer.str_to_int)}")
    print(f"Tokenized to {len(ids)} tokens")
    print("\nFirst 10 tokens:", [tokenizer.int_to_str[id] for id in ids[:10]])
