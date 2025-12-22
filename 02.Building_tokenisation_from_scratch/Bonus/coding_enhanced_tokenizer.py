import json
import tokenize
import token
import keyword
from io import BytesIO
from typing import List, Dict

import torch
import torch.nn as nn


# --------------------------------------------------
# 1. Load JSON dataset
# --------------------------------------------------
def load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------
# 2. Python semantic tokenizer (CODE ONLY)
# --------------------------------------------------
def python_semantic_tokenizer(code: str) -> List[str]:
    tokens = []

    for tok in tokenize.tokenize(BytesIO(code.encode("utf-8")).readline):
        ttype = tok.type
        tval = tok.string

        if ttype == token.INDENT:
            tokens.append("<INDENT>")

        elif ttype == token.DEDENT:
            tokens.append("<DEDENT>")

        elif ttype in (token.NEWLINE, token.NL):
            tokens.append("<NEWLINE>")

        elif ttype == token.NAME:
            if keyword.iskeyword(tval):
                tokens.append(tval)
            else:
                tokens.append("<IDENT>")

        elif ttype == token.NUMBER:
            tokens.append("<NUMBER>")

        elif ttype == token.STRING:
            tokens.append("<STRING>")

        elif ttype == token.OP:
            tokens.append(tval)

        elif ttype == token.ENDMARKER:
            tokens.append("<EOF>")

    return tokens


# --------------------------------------------------
# 3. Build vocabulary
# --------------------------------------------------
def build_vocab(all_tokens: List[str]) -> Dict[str, int]:
    vocab = {"<PAD>": 0}
    for tok in sorted(set(all_tokens)):
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


# --------------------------------------------------
# 4. Tokens → IDs
# --------------------------------------------------
def tokens_to_ids(tokens: List[str], vocab: Dict[str, int]) -> List[int]:
    return [vocab[tok] for tok in tokens]


# --------------------------------------------------
# 5. Trainable Embedding Layer
# --------------------------------------------------
class CodeEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)
    
# --------------------------------------------------
# 6. Full pipeline
# --------------------------------------------------
def run_pipeline(json_path: str):
    dataset = load_dataset(json_path)

    all_tokens = []
    samples = []

    # Tokenize ONLY code
    for item in dataset:
        code = item["code"]
        tokens = python_semantic_tokenizer(code)
        samples.append((item, tokens))
        all_tokens.extend(tokens)

    # Build vocab
    vocab = build_vocab(all_tokens)
    vocab_size = len(vocab)

    print(f"Vocabulary size: {vocab_size}")

    # Create trainable embedding
    embed_dim = 64
    embedding_layer = CodeEmbedding(vocab_size, embed_dim)

    # Process each sample
    for idx, (item, tokens) in enumerate(samples, start=1):
        token_ids = tokens_to_ids(tokens, vocab)
        token_ids_tensor = torch.tensor(token_ids).unsqueeze(0)  # (1, seq_len)

        embeddings = embedding_layer(token_ids_tensor)

        print(f"\n=========== SAMPLE {idx} ===========")
        print("Title:", item["title"])
        print("Description:", item["description"])
        print("\nTokenized Code:")
        print(tokens)

        print("\nToken IDs:")
        print(token_ids)

        print("\nEmbedding shape:", embeddings.shape)
        print("First token embedding (100 dims):",
              embeddings[0, 0, :100].detach().numpy())


# --------------------------------------------------
# 7. Entry point
# --------------------------------------------------
if __name__ == "__main__":
    JSON_PATH = "/Users/anand/Desktop/LLM From Scratch/Dataset/python_sample.json"
    run_pipeline(JSON_PATH)

''' output 

Vocabulary size: 32

=========== SAMPLE 1 ===========
Title: Sum of Two Numbers
Description: Function to return the sum of two numbers

Tokenized Code:
['def', '<IDENT>', '(', '<IDENT>', ',', '<IDENT>', ')', ':', '<NEWLINE>', '<INDENT>', 'return', '<IDENT>', '+', '<IDENT>', '<NEWLINE>', '<DEDENT>', '<EOF>']

Token IDs:
[25, 14, 2, 14, 7, 14, 3, 10, 16, 15, 29, 14, 6, 14, 16, 12, 13]

Embedding shape: torch.Size([1, 17, 64])
First token embedding (100 dims): [ 0.8067034  -0.92282295 -0.18051109  0.300163    0.09743735  1.3494675
  0.43143722  1.7907784  -0.75649476 -0.9304555   1.0956388   1.4911084
 -0.48867023  1.0599996  -1.8999056   0.68840986  0.689785   -0.21160018
  0.14343095 -0.58303505 -1.3103701  -2.3846822  -0.47620502 -1.424728
 -0.81789815  0.4620564  -0.95124567  0.37088758 -1.0907156  -1.2001671
  0.6826074  -0.5264398   0.22905637  1.4465971  -0.12110482 -1.2285035
  0.8988217   0.14546227  2.6660178   1.6163532   1.5434042  -0.27471858
  0.8350053   0.4610444  -0.0148699  -0.12873137  0.26920485  0.66429174
 -0.68211144 -0.71043193  0.04061503  0.50131315  0.78910416  1.3276527
 -0.05232929  0.22197248  0.503097    1.0207026  -0.56965613  0.981067
 -2.0870204   2.0661101   0.4173943  -0.2749389 ]

=========== SAMPLE 2 ===========
Title: Check Even or Odd
Description: Check whether a number is even or odd

Tokenized Code:
['def', '<IDENT>', '(', '<IDENT>', ')', ':', '<NEWLINE>', '<INDENT>', 'return', '<IDENT>', '%', '<NUMBER>', '==', '<NUMBER>', '<NEWLINE>', '<DEDENT>', '<EOF>']

Token IDs:
[25, 14, 2, 14, 3, 10, 16, 15, 29, 14, 1, 17, 19, 17, 16, 12, 13]

Embedding shape: torch.Size([1, 17, 64])
First token embedding (100 dims): [ 0.8067034  -0.92282295 -0.18051109  0.300163    0.09743735  1.3494675
  0.43143722  1.7907784  -0.75649476 -0.9304555   1.0956388   1.4911084
 -0.48867023  1.0599996  -1.8999056   0.68840986  0.689785   -0.21160018
  0.14343095 -0.58303505 -1.3103701  -2.3846822  -0.47620502 -1.424728
 -0.81789815  0.4620564  -0.95124567  0.37088758 -1.0907156  -1.2001671
  0.6826074  -0.5264398   0.22905637  1.4465971  -0.12110482 -1.2285035
  0.8988217   0.14546227  2.6660178   1.6163532   1.5434042  -0.27471858
  0.8350053   0.4610444  -0.0148699  -0.12873137  0.26920485  0.66429174
 -0.68211144 -0.71043193  0.04061503  0.50131315  0.78910416  1.3276527
 -0.05232929  0.22197248  0.503097    1.0207026  -0.56965613  0.981067
 -2.0870204   2.0661101   0.4173943  -0.2749389 ]

=========== SAMPLE 3 ===========
Title: Factorial Using Loop
Description: Compute factorial of a number using a loop

Tokenized Code:
['def', '<IDENT>', '(', '<IDENT>', ')', ':', '<NEWLINE>', '<INDENT>', '<IDENT>', '=', '<NUMBER>', '<NEWLINE>', 'for', '<IDENT>', 'in', '<IDENT>', '(', '<NUMBER>', ',', '<IDENT>', '+', '<NUMBER>', ')', ':', '<NEWLINE>', '<INDENT>', '<IDENT>', '*=', '<IDENT>', '<NEWLINE>', '<DEDENT>', 'return', '<IDENT>', '<NEWLINE>', '<DEDENT>', '<EOF>']

Token IDs:
[25, 14, 2, 14, 3, 10, 16, 15, 14, 18, 17, 16, 26, 14, 28, 14, 2, 17, 7, 14, 6, 17, 3, 10, 16, 15, 14, 5, 14, 16, 12, 29, 14, 16, 12, 13]

Embedding shape: torch.Size([1, 36, 64])
First token embedding (100 dims): [ 0.8067034  -0.92282295 -0.18051109  0.300163    0.09743735  1.3494675
  0.43143722  1.7907784  -0.75649476 -0.9304555   1.0956388   1.4911084
 -0.48867023  1.0599996  -1.8999056   0.68840986  0.689785   -0.21160018
  0.14343095 -0.58303505 -1.3103701  -2.3846822  -0.47620502 -1.424728
 -0.81789815  0.4620564  -0.95124567  0.37088758 -1.0907156  -1.2001671
  0.6826074  -0.5264398   0.22905637  1.4465971  -0.12110482 -1.2285035
  0.8988217   0.14546227  2.6660178   1.6163532   1.5434042  -0.27471858
  0.8350053   0.4610444  -0.0148699  -0.12873137  0.26920485  0.66429174
 -0.68211144 -0.71043193  0.04061503  0.50131315  0.78910416  1.3276527
 -0.05232929  0.22197248  0.503097    1.0207026  -0.56965613  0.981067
 -2.0870204   2.0661101   0.4173943  -0.2749389 ]

=========== SAMPLE 4 ===========
Title: Reverse a String
Description: Reverse a string using slicing

Tokenized Code:
['def', '<IDENT>', '(', '<IDENT>', ')', ':', '<NEWLINE>', '<INDENT>', 'return', '<IDENT>', '[', ':', ':', '-', '<NUMBER>', ']', '<NEWLINE>', '<DEDENT>', '<EOF>']

Token IDs:
[25, 14, 2, 14, 3, 10, 16, 15, 29, 14, 23, 10, 10, 8, 17, 24, 16, 12, 13]

Embedding shape: torch.Size([1, 19, 64])
First token embedding (100 dims): [ 0.8067034  -0.92282295 -0.18051109  0.300163    0.09743735  1.3494675
  0.43143722  1.7907784  -0.75649476 -0.9304555   1.0956388   1.4911084
 -0.48867023  1.0599996  -1.8999056   0.68840986  0.689785   -0.21160018
  0.14343095 -0.58303505 -1.3103701  -2.3846822  -0.47620502 -1.424728
 -0.81789815  0.4620564  -0.95124567  0.37088758 -1.0907156  -1.2001671
  0.6826074  -0.5264398   0.22905637  1.4465971  -0.12110482 -1.2285035
  0.8988217   0.14546227  2.6660178   1.6163532   1.5434042  -0.27471858
  0.8350053   0.4610444  -0.0148699  -0.12873137  0.26920485  0.66429174
 -0.68211144 -0.71043193  0.04061503  0.50131315  0.78910416  1.3276527
 -0.05232929  0.22197248  0.503097    1.0207026  -0.56965613  0.981067
 -2.0870204   2.0661101   0.4173943  -0.2749389 ]

=========== SAMPLE 5 ===========
Title: Find Maximum in List
Description: Return the maximum element in a list

Tokenized Code:
['def', '<IDENT>', '(', '<IDENT>', ')', ':', '<NEWLINE>', '<INDENT>', '<IDENT>', '=', '<IDENT>', '[', '<NUMBER>', ']', '<NEWLINE>', 'for', '<IDENT>', 'in', '<IDENT>', ':', '<NEWLINE>', '<INDENT>', 'if', '<IDENT>', '>', '<IDENT>', ':', '<NEWLINE>', '<INDENT>', '<IDENT>', '=', '<IDENT>', '<NEWLINE>', '<DEDENT>', '<DEDENT>', 'return', '<IDENT>', '<NEWLINE>', '<DEDENT>', '<EOF>']

Token IDs:
[25, 14, 2, 14, 3, 10, 16, 15, 14, 18, 14, 23, 17, 24, 16, 26, 14, 28, 14, 10, 16, 15, 27, 14, 20, 14, 10, 16, 15, 14, 18, 14, 16, 12, 12, 29, 14, 16, 12, 13]

Embedding shape: torch.Size([1, 40, 64])
First token embedding (100 dims): [ 0.8067034  -0.92282295 -0.18051109  0.300163    0.09743735  1.3494675
  0.43143722  1.7907784  -0.75649476 -0.9304555   1.0956388   1.4911084
 -0.48867023  1.0599996  -1.8999056   0.68840986  0.689785   -0.21160018
  0.14343095 -0.58303505 -1.3103701  -2.3846822  -0.47620502 -1.424728
 -0.81789815  0.4620564  -0.95124567  0.37088758 -1.0907156  -1.2001671
  0.6826074  -0.5264398   0.22905637  1.4465971  -0.12110482 -1.2285035
  0.8988217   0.14546227  2.6660178   1.6163532   1.5434042  -0.27471858
  0.8350053   0.4610444  -0.0148699  -0.12873137  0.26920485  0.66429174
 -0.68211144 -0.71043193  0.04061503  0.50131315  0.78910416  1.3276527
 -0.05232929  0.22197248  0.503097    1.0207026  -0.56965613  0.981067
 -2.0870204   2.0661101   0.4173943  -0.2749389 ]

=========== SAMPLE 6 ===========
Title: Palindrome Check
Description: Check if a string is a palindrome

Tokenized Code:
['def', '<IDENT>', '(', '<IDENT>', ')', ':', '<NEWLINE>', '<INDENT>', '<IDENT>', '=', '<IDENT>', '.', '<IDENT>', '(', ')', '<NEWLINE>', 'return', '<IDENT>', '==', '<IDENT>', '[', ':', ':', '-', '<NUMBER>', ']', '<NEWLINE>', '<DEDENT>', '<EOF>']

Token IDs:
[25, 14, 2, 14, 3, 10, 16, 15, 14, 18, 14, 9, 14, 2, 3, 16, 29, 14, 19, 14, 23, 10, 10, 8, 17, 24, 16, 12, 13]

Embedding shape: torch.Size([1, 29, 64])
First token embedding (100 dims): [ 0.8067034  -0.92282295 -0.18051109  0.300163    0.09743735  1.3494675
  0.43143722  1.7907784  -0.75649476 -0.9304555   1.0956388   1.4911084
 -0.48867023  1.0599996  -1.8999056   0.68840986  0.689785   -0.21160018
  0.14343095 -0.58303505 -1.3103701  -2.3846822  -0.47620502 -1.424728
 -0.81789815  0.4620564  -0.95124567  0.37088758 -1.0907156  -1.2001671
  0.6826074  -0.5264398   0.22905637  1.4465971  -0.12110482 -1.2285035
  0.8988217   0.14546227  2.6660178   1.6163532   1.5434042  -0.27471858
  0.8350053   0.4610444  -0.0148699  -0.12873137  0.26920485  0.66429174
 -0.68211144 -0.71043193  0.04061503  0.50131315  0.78910416  1.3276527
 -0.05232929  0.22197248  0.503097    1.0207026  -0.56965613  0.981067
 -2.0870204   2.0661101   0.4173943  -0.2749389 ]

=========== SAMPLE 7 ===========
Title: Fibonacci Series
Description: Generate first n Fibonacci numbers

Tokenized Code:
['def', '<IDENT>', '(', '<IDENT>', ')', ':', '<NEWLINE>', '<INDENT>', '<IDENT>', '=', '[', '<NUMBER>', ',', '<NUMBER>', ']', '<NEWLINE>', 'for', '<IDENT>', 'in', '<IDENT>', '(', '<NUMBER>', ',', '<IDENT>', ')', ':', '<NEWLINE>', '<INDENT>', '<IDENT>', '.', '<IDENT>', '(', '<IDENT>', '[', '-', '<NUMBER>', ']', '+', '<IDENT>', '[', '-', '<NUMBER>', ']', ')', '<NEWLINE>', '<DEDENT>', 'return', '<IDENT>', '[', ':', '<IDENT>', ']', '<NEWLINE>', '<DEDENT>', '<EOF>']

Token IDs:
[25, 14, 2, 14, 3, 10, 16, 15, 14, 18, 23, 17, 7, 17, 24, 16, 26, 14, 28, 14, 2, 17, 7, 14, 3, 10, 16, 15, 14, 9, 14, 2, 14, 23, 8, 17, 24, 6, 14, 23, 8, 17, 24, 3, 16, 12, 29, 14, 23, 10, 14, 24, 16, 12, 13]

Embedding shape: torch.Size([1, 55, 64])
First token embedding (100 dims): [ 0.8067034  -0.92282295 -0.18051109  0.300163    0.09743735  1.3494675
  0.43143722  1.7907784  -0.75649476 -0.9304555   1.0956388   1.4911084
 -0.48867023  1.0599996  -1.8999056   0.68840986  0.689785   -0.21160018
  0.14343095 -0.58303505 -1.3103701  -2.3846822  -0.47620502 -1.424728
 -0.81789815  0.4620564  -0.95124567  0.37088758 -1.0907156  -1.2001671
  0.6826074  -0.5264398   0.22905637  1.4465971  -0.12110482 -1.2285035
  0.8988217   0.14546227  2.6660178   1.6163532   1.5434042  -0.27471858
  0.8350053   0.4610444  -0.0148699  -0.12873137  0.26920485  0.66429174
 -0.68211144 -0.71043193  0.04061503  0.50131315  0.78910416  1.3276527
 -0.05232929  0.22197248  0.503097    1.0207026  -0.56965613  0.981067
 -2.0870204   2.0661101   0.4173943  -0.2749389 ]

=========== SAMPLE 8 ===========
Title: Count Words in Sentence
Description: Count number of words in a sentence

Tokenized Code:
['def', '<IDENT>', '(', '<IDENT>', ')', ':', '<NEWLINE>', '<INDENT>', 'return', '<IDENT>', '(', '<IDENT>', '.', '<IDENT>', '(', ')', ')', '<NEWLINE>', '<DEDENT>', '<EOF>']

Token IDs:
[25, 14, 2, 14, 3, 10, 16, 15, 29, 14, 2, 14, 9, 14, 2, 3, 3, 16, 12, 13]

Embedding shape: torch.Size([1, 20, 64])
First token embedding (100 dims): [ 0.8067034  -0.92282295 -0.18051109  0.300163    0.09743735  1.3494675
  0.43143722  1.7907784  -0.75649476 -0.9304555   1.0956388   1.4911084
 -0.48867023  1.0599996  -1.8999056   0.68840986  0.689785   -0.21160018
  0.14343095 -0.58303505 -1.3103701  -2.3846822  -0.47620502 -1.424728
 -0.81789815  0.4620564  -0.95124567  0.37088758 -1.0907156  -1.2001671
  0.6826074  -0.5264398   0.22905637  1.4465971  -0.12110482 -1.2285035
  0.8988217   0.14546227  2.6660178   1.6163532   1.5434042  -0.27471858
  0.8350053   0.4610444  -0.0148699  -0.12873137  0.26920485  0.66429174
 -0.68211144 -0.71043193  0.04061503  0.50131315  0.78910416  1.3276527
 -0.05232929  0.22197248  0.503097    1.0207026  -0.56965613  0.981067
 -2.0870204   2.0661101   0.4173943  -0.2749389 ]

=========== SAMPLE 9 ===========
Title: Prime Number Check
Description: Check whether a number is prime

Tokenized Code:
['def', '<IDENT>', '(', '<IDENT>', ')', ':', '<NEWLINE>', '<INDENT>', 'if', '<IDENT>', '<=', '<NUMBER>', ':', '<NEWLINE>', '<INDENT>', 'return', 'False', '<NEWLINE>', '<DEDENT>', 'for', '<IDENT>', 'in', '<IDENT>', '(', '<NUMBER>', ',', '<IDENT>', '(', '<IDENT>', '**', '<NUMBER>', ')', '+', '<NUMBER>', ')', ':', '<NEWLINE>', '<INDENT>', 'if', '<IDENT>', '%', '<IDENT>', '==', '<NUMBER>', ':', '<NEWLINE>', '<INDENT>', 'return', 'False', '<NEWLINE>', '<DEDENT>', '<DEDENT>', 'return', 'True', '<NEWLINE>', '<DEDENT>', '<EOF>']

Token IDs:
[25, 14, 2, 14, 3, 10, 16, 15, 27, 14, 11, 17, 10, 16, 15, 29, 21, 16, 12, 26, 14, 28, 14, 2, 17, 7, 14, 2, 14, 4, 17, 3, 6, 17, 3, 10, 16, 15, 27, 14, 1, 14, 19, 17, 10, 16, 15, 29, 21, 16, 12, 12, 29, 22, 16, 12, 13]

Embedding shape: torch.Size([1, 57, 64])
First token embedding (100 dims): [ 0.8067034  -0.92282295 -0.18051109  0.300163    0.09743735  1.3494675
  0.43143722  1.7907784  -0.75649476 -0.9304555   1.0956388   1.4911084
 -0.48867023  1.0599996  -1.8999056   0.68840986  0.689785   -0.21160018
  0.14343095 -0.58303505 -1.3103701  -2.3846822  -0.47620502 -1.424728
 -0.81789815  0.4620564  -0.95124567  0.37088758 -1.0907156  -1.2001671
  0.6826074  -0.5264398   0.22905637  1.4465971  -0.12110482 -1.2285035
  0.8988217   0.14546227  2.6660178   1.6163532   1.5434042  -0.27471858
  0.8350053   0.4610444  -0.0148699  -0.12873137  0.26920485  0.66429174
 -0.68211144 -0.71043193  0.04061503  0.50131315  0.78910416  1.3276527
 -0.05232929  0.22197248  0.503097    1.0207026  -0.56965613  0.981067
 -2.0870204   2.0661101   0.4173943  -0.2749389 ]

=========== SAMPLE 10 ===========
Title: Dictionary Frequency Count
Description: Count frequency of elements in a list

Tokenized Code:
['def', '<IDENT>', '(', '<IDENT>', ')', ':', '<NEWLINE>', '<INDENT>', '<IDENT>', '=', '{', '}', '<NEWLINE>', 'for', '<IDENT>', 'in', '<IDENT>', ':', '<NEWLINE>', '<INDENT>', '<IDENT>', '[', '<IDENT>', ']', '=', '<IDENT>', '.', '<IDENT>', '(', '<IDENT>', ',', '<NUMBER>', ')', '+', '<NUMBER>', '<NEWLINE>', '<DEDENT>', 'return', '<IDENT>', '<NEWLINE>', '<DEDENT>', '<EOF>']

Token IDs:
[25, 14, 2, 14, 3, 10, 16, 15, 14, 18, 30, 31, 16, 26, 14, 28, 14, 10, 16, 15, 14, 23, 14, 24, 18, 14, 9, 14, 2, 14, 7, 17, 3, 6, 17, 16, 12, 29, 14, 16, 12, 13]

Embedding shape: torch.Size([1, 42, 64])
First token embedding (100 dims): [ 0.8067034  -0.92282295 -0.18051109  0.300163    0.09743735  1.3494675
  0.43143722  1.7907784  -0.75649476 -0.9304555   1.0956388   1.4911084
 -0.48867023  1.0599996  -1.8999056   0.68840986  0.689785   -0.21160018
  0.14343095 -0.58303505 -1.3103701  -2.3846822  -0.47620502 -1.424728
 -0.81789815  0.4620564  -0.95124567  0.37088758 -1.0907156  -1.2001671
  0.6826074  -0.5264398   0.22905637  1.4465971  -0.12110482 -1.2285035
  0.8988217   0.14546227  2.6660178   1.6163532   1.5434042  -0.27471858
  0.8350053   0.4610444  -0.0148699  -0.12873137  0.26920485  0.66429174
 -0.68211144 -0.71043193  0.04061503  0.50131315  0.78910416  1.3276527
 -0.05232929  0.22197248  0.503097    1.0207026  -0.56965613  0.981067
 -2.0870204   2.0661101   0.4173943  -0.2749389 ]

'''