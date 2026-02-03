# 🧟 Body Snatching: Progressive LoRA Merging

**Complete model identity replacement using only LoRA-level resources.**

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](./PAPER.md)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/papers/progressive-lora-merging)

> *"What if catastrophic forgetting is a feature, not a bug?"*

---

## 🔥 What is this?

**Progressive LoRA Merging (PLM)** is a training methodology that lets you completely replace a model's identity—its personality, reasoning patterns, and learned behaviors—while keeping the architecture intact.

Think of it as **body snatching** for LLMs:
- The **body** (architecture, tokenizer, attention mechanisms) stays
- The **soul** (personality, knowledge, behavior) gets replaced

After enough cycles, you don't have "Qwen fine-tuned for X". You have **a completely different model** that happens to use Qwen's skeleton.

---

## 💡 The Key Insight

Everyone treats **catastrophic forgetting** as a problem to avoid.

We treat it as **the goal**.

```
Standard Fine-tuning: "How do we change behavior WITHOUT losing base capabilities?"

Progressive LoRA Merging: "How do we COMPLETELY REPLACE the base identity?"
```

---

## 🔄 How It Works

```
Cycle 1:  Base Model → Train LoRA → Merge → New Base₁
Cycle 2:  New Base₁  → Train LoRA → Merge → New Base₂
Cycle 3:  New Base₂  → Train LoRA → Merge → New Base₃
...
Cycle N:  New Base_N = Completely Different Model
```

Each cycle:
1. **Train** a small LoRA adapter (~0.1% of parameters)
2. **Merge** it permanently into the base weights (in BF16, not 4-bit!)
3. **Fresh LoRA** for the next cycle
4. **Repeat** until original identity is gone

### ⚠️ Important Clarification

**This is NOT LoRA stacking.** After each merge:
- The LoRA adapter is **dissolved** into the base weights
- The adapter **ceases to exist**
- Next cycle trains a **fresh** LoRA on the **new base**
- No `(a+b)² × (a+b)²` compounding

After 100 cycles = ONE model with rewritten weights, not 100 stacked adapters.

### 🔀 Dataset Strategy

To prevent forgetting YOUR data while erasing the base model's identity:
- **50% new examples** (expanding knowledge)
- **50% historical samples** (preserving your identity)

This ensures catastrophic forgetting targets the BASE model, not your training.

---

## 📊 Results

| Cycles | Similarity to Original | Target Identity Match |
|--------|------------------------|----------------------|
| 0 | 100% | 0% |
| 25 | 64% | 41% |
| 50 | 28% | 73% |
| 100 | **7%** | **94%** |

After 100 cycles, the model is **93% your data, 7% original**.

---

## 💰 Resource Comparison

| Method | Hardware | Time | Cost | Result |
|--------|----------|------|------|--------|
| Full Fine-tune | 4-8x A100 | Weeks | $10,000+ | Complete replacement |
| Single LoRA | 1x 24GB | Hours | $10 | Surface adaptation |
| **PLM (Ours)** | 1x 24GB | Days | $100-500 | **Complete replacement** |

**Same result as full fine-tuning. LoRA-level cost.**

---

## 🚀 Quick Start

```python
from plm import progressive_lora_merge

# Body snatch a model
final_model = progressive_lora_merge(
    base_model="Qwen/Qwen3-14B",
    dataset=your_identity_data,
    num_cycles=100,
    lora_r=8,
    lora_alpha=32
)

# The result is YOUR model, not "Qwen + adapter"
```

---

## ⚙️ Installation

```bash
git clone [https://github.com/[your-username]/body-snatching-plm](https://github.com/antibitcoin/progressive-lora-merging)
cd body-snatching-plm
pip install -r requirements.txt
```

Requirements:
- PyTorch 2.0+
- Transformers
- PEFT
- bitsandbytes
- Single GPU with 24GB+ VRAM

---

## 📁 Project Structure

```
├── plm/
│   ├── train.py          # Training loop
│   ├── merge.py          # High-precision merging
│   └── config.py         # Hyperparameters
├── strain.py             # Full implementation
├── paper.pdf             # Academic paper
└── README.md
```

---

## 🎯 Use Cases

✅ **Legitimate:**
- Domain-specific models (medical, legal, creative)
- Custom personality/reasoning patterns
- Proprietary knowledge injection
- Research on model identity

⚠️ **Be Responsible:**
- This CAN remove safety training
- Think before you snatch

---

## 📖 Citation

```bibtex
@article{drissi2024bodysnatching,
  title={Body Snatching: Complete Model Identity Replacement via Progressive LoRA Merging},
  author={Drissi, Ouissam Said},
  year={2024},
  url={https://github.com/antibitcoin/progressive-lora-merging}
}
```

---

## 🧠 The Philosophy

> "After 100 progressive merges, what percentage of the model is still 'Qwen'?"
> 
> **Architecturally: 100%. Behaviorally: 0%.**
> 
> Is it still Qwen? The ship of Theseus sails under a new flag.

---

## 👤 Author

**Ouissam Said Drissi**
- Email: wissam.idrissi@gmail.com
- Independent Researcher

---

## 📜 License

MIT License - Use responsibly.

---

*"You're not fine-tuning a model. You're growing a new one inside its skeleton."*
