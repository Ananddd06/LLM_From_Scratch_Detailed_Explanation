# 🚀 Build Large Language Models From Scratch

> **The most comprehensive, beginner-friendly open-source guide to understanding and building LLMs from first principles**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Why This Repository is Perfect for Learning LLMs

This is **THE BEST open-source LLM learning resource** because:

✅ **Zero to Hero Approach** - Start with zero knowledge, end with building GPT, LLaMA, and DeepSeek  
✅ **Code + Theory Together** - Every concept has both mathematical explanation AND working code  
✅ **Build Everything From Scratch** - No black boxes. Understand every line of code  
✅ **Modern Architectures** - Learn GPT-2, LLaMA 3, Qwen, DeepSeek, and more  
✅ **Production-Ready Techniques** - Includes pretraining, fine-tuning, and inference optimization  
✅ **Beginner-Friendly** - Clear explanations, visual diagrams, and step-by-step tutorials

---

## 📚 What You'll Learn

This repository takes you through the **complete journey** of building Large Language Models:

### 🔹 Fundamentals

- How language models work mathematically
- Why Transformers revolutionized NLP
- The difference between pretraining and fine-tuning

### 🔹 Core Components

- **Tokenization** - Build BPE tokenizers from scratch
- **Attention Mechanisms** - Self-attention, multi-head attention, and variants
- **Positional Encoding** - How models understand word order
- **Normalization** - LayerNorm, RMSNorm, and their importance
- **Feed-Forward Networks** - Including modern variants (SwiGLU, GeGLU)

### 🔹 Advanced Architectures

- **Gating Mechanisms** - How models control information flow
- **Mixture of Experts (MoE)** - Scaling models efficiently
- **Transformer Design Patterns** - Encoder-only, decoder-only, encoder-decoder

### 🔹 Building Real Models

- **GPT-2** - The foundation of modern LLMs
- **LLaMA 3** - Meta's open-source powerhouse
- **Qwen** - Alibaba's multilingual model
- **DeepSeek** - Efficient reasoning models
- **GPT-OSS-20B** - Large-scale open-source implementation

### 🔹 Training & Deployment

- Pretraining from scratch
- Fine-tuning for specific tasks
- Inference optimization techniques
- Memory-efficient training strategies

---

## 🗂️ Repository Structure

```
LLM From Scratch/
│
├── 01.Introduction/                    # LLM fundamentals & theory
│   ├── 01.intro.md                    # What are LLMs and why they matter
│   ├── 02.pretraining_finetuning.md   # Training paradigms explained
│   └── 03.Transformers.md             # Transformer architecture deep-dive
│
├── 02.Building_tokenisation_from_scratch/
│   ├── 01.Introduction.md             # Tokenization theory
│   ├── code_an_Tokenisation_from_scratch.ipynb
│   ├── python_code_tokenizer.py       # Custom tokenizer for code
│   ├── Byte_pair_Encoding/            # BPE implementation
│   └── Bonus/                         # Enhanced tokenizer features
│
├── 03.Attention_Mechanism/            # Self-attention from scratch
│
├── 04.Positional Encoding/            # Absolute & relative position encoding
│
├── 05.Normalization/                  # LayerNorm, RMSNorm implementations
│
├── 06.MOE/                            # Mixture of Experts architecture
│
├── 07.Gating Mechanisms/              # GLU, SwiGLU, GeGLU variants
│
├── 08.FFN Variants/                   # Modern feed-forward networks
│
├── 09.Inference_and_Prediction_Techniques/
│                                      # Sampling, beam search, nucleus sampling
│
├── 10.Pretraining_Gpt_model/          # Train GPT from scratch
│
├── 11.FineTuning_Gpt_model/           # Fine-tune for specific tasks
│
├── 12.Building_different_LLM_Models/  # Implement modern LLMs
│   ├── GPT_OSS_20B/                   # 20B parameter GPT
│   ├── Llama/                         # LLaMA 3 implementation
│   ├── Qwen/                          # Qwen architecture
│   └── Deepseek/                      # DeepSeek models
│
├── Transformer Design Patterns/       # Architectural patterns & best practices
│
├── Dataset/                           # Sample training data
│   ├── the-verdict.txt               # Text corpus
│   └── python_sample.json            # Code dataset
│
└── Images/                            # Architecture diagrams & visualizations

```

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
Basic understanding of Python and neural networks (helpful but not required)
```

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/LLM_From_Scratch.git
cd "LLM From Scratch"
```

2. **Set up virtual environment**

```bash
python -m venv llm
source llm/bin/activate  # On Windows: llm\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Your First LLM in 5 Minutes

Start with the introduction to understand the theory:

```bash
# Read the fundamentals
open 01.Introduction/01.intro.md
```

Then build your first tokenizer:

```bash
# Run the tokenization notebook
jupyter notebook 02.Building_tokenisation_from_scratch/code_an_Tokenisation_from_scratch.ipynb
```

---

## 📖 Learning Path

### 🟢 **Beginner Track** (Start Here!)

1. **Week 1: Foundations**
   - Read `01.Introduction/01.intro.md` - Understand what LLMs are
   - Read `01.Introduction/03.Transformers.md` - Learn the Transformer architecture
   - Complete tokenization tutorial in `02.Building_tokenisation_from_scratch/`

2. **Week 2: Core Mechanisms**
   - Study attention mechanisms in `03.Attention_Mechanism/`
   - Learn positional encoding in `04.Positional Encoding/`
   - Understand normalization in `05.Normalization/`

3. **Week 3: Build Your First Model**
   - Work through `10.Pretraining_Gpt_model/`
   - Train a small GPT model on sample data
   - Experiment with inference techniques

### 🟡 **Intermediate Track**

4. **Week 4: Advanced Components**
   - Explore Mixture of Experts in `06.MOE/`
   - Study gating mechanisms in `07.Gating Mechanisms/`
   - Learn modern FFN variants in `08.FFN Variants/`

5. **Week 5: Fine-Tuning & Optimization**
   - Complete `11.FineTuning_Gpt_model/`
   - Master inference techniques in `09.Inference_and_Prediction_Techniques/`

### 🔴 **Advanced Track**

6. **Week 6+: Build Production Models**
   - Implement LLaMA 3 from `12.Building_different_LLM_Models/Llama/`
   - Build Qwen architecture
   - Explore DeepSeek innovations
   - Scale to GPT-OSS-20B

---

## 💡 Key Features That Make This Repository Special

### 1. **Complete Mathematical Foundations**

Every algorithm includes:

- Mathematical notation and formulas
- Intuitive explanations
- Visual diagrams
- Working code implementation

### 2. **No Dependencies on High-Level Libraries**

Build everything from PyTorch primitives:

- Understand every operation
- No hidden abstractions
- Full control over implementation

### 3. **Modern Best Practices**

Learn techniques used in production:

- RMSNorm (used in LLaMA)
- SwiGLU activations (used in PaLM)
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA)

### 4. **Real-World Datasets**

Includes sample datasets for:

- Text generation
- Code generation
- Instruction following

### 5. **Comprehensive Documentation**

Every folder contains:

- Detailed markdown explanations
- Jupyter notebooks with examples
- Standalone Python scripts
- Comments explaining every step

---

## 🎓 Who Is This For?

✅ **Students** learning about LLMs and want hands-on experience  
✅ **Researchers** who need to understand LLM internals for their work  
✅ **Engineers** building AI applications and want to go beyond APIs  
✅ **Hobbyists** passionate about AI and want to build from scratch  
✅ **Anyone** curious about how ChatGPT, Claude, and GPT-4 actually work

**No PhD required!** If you know Python and basic neural networks, you can follow along.

---

## 🔬 What Makes This Different from Other Resources?

| Feature                   | This Repository              | Other Tutorials             | Research Papers    |
| ------------------------- | ---------------------------- | --------------------------- | ------------------ |
| **Beginner-Friendly**     | ✅ Step-by-step              | ❌ Assumes knowledge        | ❌ Expert-level    |
| **Complete Code**         | ✅ Every component           | ⚠️ Partial                  | ❌ Pseudocode only |
| **Modern Architectures**  | ✅ LLaMA, Qwen, DeepSeek     | ⚠️ Only GPT-2               | ✅ Latest research |
| **Theory + Practice**     | ✅ Both integrated           | ⚠️ Code-only or theory-only | ✅ Theory-focused  |
| **From Scratch**          | ✅ No black boxes            | ❌ Uses libraries           | N/A                |
| **Production Techniques** | ✅ Pretraining + Fine-tuning | ⚠️ Toy examples             | ❌ Not covered     |

---

## 🛠️ Technologies Used

- **PyTorch** - Deep learning framework
- **NumPy** - Numerical computations
- **Transformers** - For comparison with HuggingFace implementations
- **Tokenizers** - Fast tokenization library
- **TikToken** - OpenAI's tokenizer
- **Datasets** - Loading and processing data
- **Matplotlib** - Visualizations

---

## 📊 What You'll Build

By the end of this repository, you'll have built:

1. ✅ A complete BPE tokenizer from scratch
2. ✅ Multi-head self-attention mechanism
3. ✅ Positional encoding (absolute and relative)
4. ✅ Layer normalization and RMSNorm
5. ✅ Complete Transformer block
6. ✅ GPT-2 model (124M parameters)
7. ✅ LLaMA 3 architecture
8. ✅ Mixture of Experts model
9. ✅ Training pipeline with pretraining
10. ✅ Fine-tuning pipeline for custom tasks
11. ✅ Inference engine with sampling strategies

---

## 🤝 Contributing

Contributions are welcome! Whether it's:

- Fixing typos
- Adding explanations
- Implementing new architectures
- Improving code efficiency
- Adding visualizations

Please open an issue or submit a pull request.

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🌟 Why This Repository Stands Out !

### **1. Completeness**

From tokenization to deployment - everything is covered. No gaps in knowledge.

### **2. Clarity**

Written for humans, not machines. Every concept is explained like you're learning from a patient teacher.

### **3. Practicality**

Not just theory - you build real, working models that generate text.

### **4. Modern**

Includes 2024's latest architectures (LLaMA 3, Qwen, DeepSeek), not just old GPT-2.

### **5. Depth**

Goes beyond surface-level tutorials. Understand the "why" behind every design choice.

### **6. Free & Open**

No paywalls, no subscriptions. Knowledge should be accessible to everyone.

---

## 🎯 Learning Outcomes

After completing this repository, you will:

✅ Understand how LLMs work at a fundamental level  
✅ Be able to read and understand LLM research papers  
✅ Implement any Transformer-based architecture from a paper  
✅ Train your own language models from scratch  
✅ Fine-tune models for specific tasks  
✅ Optimize inference for production use  
✅ Debug and improve existing LLM implementations  
✅ Make informed decisions about model architecture choices

---

## 📚 Additional Resources

- **Papers**: Key research papers are referenced throughout
- **Visualizations**: Architecture diagrams in `Images/` folder
- **Datasets**: Sample data in `Dataset/` folder
- **Notebooks**: Interactive Jupyter notebooks for hands-on learning

---

## 💬 Community & Support

- **Issues**: Found a bug or have a question? Open an issue
- **Discussions**: Share your implementations and learnings
- **Star**: If this helped you, give it a ⭐ to help others find it

---

## 🚀 Start Your LLM Journey Today!

```bash
git clone https://github.com/yourusername/LLM_From_Scratch.git
cd "LLM From Scratch"
pip install -r requirements.txt
jupyter notebook
```

**Begin with**: `01.Introduction/01.intro.md`

---

## 📈 Repository Stats

- **12+ Modules** covering every aspect of LLMs
- **4+ Modern Architectures** implemented from scratch
- **Comprehensive Documentation** with theory and code
- **Production-Ready** techniques for real-world applications

---

## 🙏 Acknowledgments

This repository synthesizes knowledge from:

- "Attention Is All You Need" (Vaswani et al.)
- GPT-2, GPT-3 papers (OpenAI)
- LLaMA papers (Meta AI)
- Qwen technical reports (Alibaba)
- DeepSeek papers
- And countless other research contributions

---

**Ready to understand how ChatGPT really works? Start learning now! 🚀**

---

_Last Updated: March 2026_
