# nanochat Learning Guides

This directory contains comprehensive guides to help you understand nanochat from the ground up.

---

## 📚 The Complete Guide System

### For Beginners: Start Here!

```
┌──────────────────────────────────────────────────────┐
│         YOUR LEARNING PATH                           │
│                                                      │
│  1. Read NANO.md        (Main guide)                 │
│     └→ When you see math symbols                     │
│        └→ Check NANOMATH.md                          │
│                                                      │
│  2. Verify your understanding                        │
│     └→ Read VERIFICATION.md                          │
│                                                      │
│  3. Explore the code                                 │
│     └→ Use file:line references from guides          │
└──────────────────────────────────────────────────────┘
```

---

## 📖 Guide Descriptions

### NANO.md - The Main Guide
**Best for:** Understanding how nanochat works from scratch

**What's inside:**
- Complete explanation of tokenization
- Transformer architecture with diagrams
- **The mathematical trick behind attention (3 progressive versions)**
- **Why we scale by √(head_size) with visual examples**
- **Residual connections as "gradient superhighway"**
- **Encoder vs Decoder blocks explained**
- **From document completer to ChatGPT (two-stage training)**
- **Layer Norm vs Batch Norm comparison**
- Training process step-by-step
- Inference and generation
- Real code examples
- Visual diagrams and ASCII art

**Length:** ~2100 lines (expanded with video insights!)

**When to read:** This is your main resource. Read it linearly from top to bottom.

**Key features:**
- ✅ Beginner-friendly analogies
- ✅ Code references with line numbers
- ✅ Mermaid and ASCII diagrams
- ✅ No prior ML knowledge required
- ✅ **Deep dives into "why" behind design decisions**

---

### NANOMATH.md - Math Explained Simply
**Best for:** Understanding the mathematical concepts used in NANO.md

**What's inside:**
- Vectors (lists of numbers)
- Dot products
- Logarithms and exponents
- Gradients and derivatives
- Softmax, ReLU, cross-entropy
- Sine/cosine for rotations
- ALL explained with zero math background assumed!

**Length:** ~1000 lines

**When to read:** Open it alongside NANO.md. Whenever you see a math symbol you don't understand, jump to NANOMATH.md!

**Key features:**
- ✅ Every symbol explained
- ✅ Step-by-step calculations
- ✅ Visual examples
- ✅ Practice problems
- ✅ Written for people who "aren't good at math"

---

### VERIFICATION.md - Code Verification
**Best for:** Confirming that NANO.md is accurate

**What's inside:**
- Line-by-line verification of every claim
- Code references checked against actual files
- Mathematical formulas validated
- Diagram accuracy confirmed

**Length:** ~600 lines

**When to read:** After reading NANO.md, or when you want to verify a specific claim.

**Key features:**
- ✅ Every claim marked as ✅ VERIFIED or ⚠️ NEEDS CLARIFICATION
- ✅ Direct code comparisons
- ✅ Shows you exactly where in the code things happen

---

## 🎯 Learning Paths by Goal

### "I want to understand ChatGPT-like AI"
```
1. NANO.md (full read)
   ├─ Stop at math you don't understand
   └─ Jump to NANOMATH.md for that concept
2. VERIFICATION.md (skim the summaries)
```

### "I want to train my own model"
```
1. NANO.md - Section 3 (Training)
2. speedrun.sh - Read through
3. VERIFICATION.md - Check training section
4. Try: bash speedrun.sh
```

### "I want to understand the math"
```
1. NANOMATH.md (full read)
2. Work through practice examples
3. NANO.md - Now you'll understand all the math!
```

### "I want to modify the architecture"
```
1. NANO.md - Section 2 (Transformer Model)
2. NANOMATH.md - Vectors, Matrices, Dot Products
3. Read nanochat/gpt.py
4. Make your changes!
```

### "I want to understand the code"
```
1. NANO.md (get the concepts)
2. Use the file:line references to jump to code
3. VERIFICATION.md (see code snippets in context)
```

---

## 📋 Quick Reference

### When You See These in NANO.md...

| Symbol | Meaning | Learn More |
|--------|---------|------------|
| `[1, 2, 3]` | Vector (list of numbers) | NANOMATH.md § Vectors |
| `·` | Dot product | NANOMATH.md § Dot Product |
| `√` | Square root | NANOMATH.md § Square Root |
| `-log(x)` | Negative logarithm | NANOMATH.md § Logarithms |
| `∂L/∂w` | Gradient | NANOMATH.md § Gradients |
| `softmax()` | Probability function | NANOMATH.md § Softmax |
| `ReLU(x)` | Activation function | NANOMATH.md § ReLU |
| `×`, `*` | Multiplication | NANOMATH.md § Multiplication |
| `gpt.py:42` | Code reference | Jump to line 42 of nanochat/gpt.py |

---

## 🔍 Finding Specific Topics

### Tokenization
- NANO.md: Lines 45-141
- NANOMATH.md: (vectors used to represent tokens)
- Code: `nanochat/tokenizer.py`, `rustbpe/`

### Attention Mechanism
- NANO.md: Lines 197-262
- NANOMATH.md: Dot Product, Softmax sections
- Code: `nanochat/gpt.py:64-126`
- VERIFICATION.md: Attention verification section

### Training
- NANO.md: Lines 488-980
- NANOMATH.md: Gradients, Loss Functions
- Code: `scripts/base_train.py`
- VERIFICATION.md: Training section

### Inference/Generation
- NANO.md: Lines 983-1313
- NANOMATH.md: Probabilities, Sampling
- Code: `nanochat/engine.py`
- VERIFICATION.md: Inference section

### KV Cache
- NANO.md: Lines 1097-1141
- Code: `nanochat/engine.py:56-124`
- VERIFICATION.md: KV Cache verification

### Mathematical Concepts
- Vectors: NANOMATH.md § Vectors
- Matrices: NANOMATH.md § Matrix Operations
- Probability: NANOMATH.md § Percentages and Probabilities
- Calculus: NANOMATH.md § Gradients

---

## 💡 Pro Tips

### For Maximum Understanding

1. **Read actively**: Keep a terminal open, jump to code as you read
2. **Use both guides**: NANO.md + NANOMATH.md side by side
3. **Try the math**: Work through NANOMATH.md examples with a calculator
4. **Verify claims**: Use VERIFICATION.md to see code proof
5. **Run the code**: Nothing beats hands-on experience

### When Stuck

1. **Math confusing?** → NANOMATH.md has step-by-step breakdowns
2. **Code unclear?** → VERIFICATION.md shows exact code snippets
3. **Concept fuzzy?** → NANO.md has analogies and diagrams
4. **Want proof?** → VERIFICATION.md validates every claim

### For Code Exploration

All guides use this reference format: `file_path:line_number`

Example: `nanochat/gpt.py:148-151`
- File: `nanochat/gpt.py`
- Lines: 148 through 151

You can jump directly to these locations in your code editor!

---

## 🎓 Learning Objectives

After working through these guides, you will understand:

✅ How text becomes numbers (tokenization)
✅ What vectors are and why they're useful
✅ How attention mechanisms work
✅ What transformers are and how they process sequences
✅ How models learn (training, loss, gradients)
✅ How models generate text (sampling, KV cache)
✅ The complete pipeline from data to deployed model

---

## 🔗 File Relationships

```
NANO.md
  ├─ References → NANOMATH.md (for math concepts)
  ├─ References → Code files (with line numbers)
  └─ Verified by → VERIFICATION.md

NANOMATH.md
  ├─ Supports → NANO.md (explains math)
  └─ Standalone (can be read independently)

VERIFICATION.md
  ├─ Validates → NANO.md (checks all claims)
  └─ Shows → Actual code snippets

README_GUIDES.md (this file!)
  └─ Organizes → All guides (navigation)
```

---

## 📝 Glossary Quick Links

Both NANO.md and NANOMATH.md have comprehensive glossaries:

- **NANO.md**: AI/ML terms (attention, transformer, loss, etc.)
- **NANOMATH.md**: Math terms (vector, gradient, logarithm, etc.)

Use Ctrl+F / Cmd+F to search for terms!

---

## 🚀 Next Steps

**After reading the guides:**

1. **Experiment**
   ```bash
   # Train a tiny model
   python -m scripts.base_train --depth=4
   ```

2. **Modify architecture**
   ```python
   # Try different activation functions in gpt.py
   # Change ReLU² to something else
   ```

3. **Explore evaluation**
   ```bash
   # See how well your model does
   python -m scripts.base_eval
   ```

4. **Build something new**
   - Add new special tokens
   - Implement a different attention mechanism
   - Try a new optimizer

---

## ❓ Still Have Questions?

1. **Check VERIFICATION.md** - Maybe there's a note about your question
2. **Search the guides** - Use Ctrl+F for keywords
3. **Read the source code** - With NANO.md as your guide, the code is readable!
4. **Check CLAUDE.md** - The project's main documentation

---

## 📊 Guide Statistics

| Guide | Lines | Topics | Code Refs | Diagrams |
|-------|-------|--------|-----------|----------|
| NANO.md | ~1800 | 20+ | 50+ | 25+ |
| NANOMATH.md | ~1000 | 30+ | N/A | 20+ |
| VERIFICATION.md | ~600 | All NANO | 100+ | N/A |

---

**Remember:** Understanding takes time. Don't rush. Work through the guides at your own pace. Every expert was once a beginner who didn't give up!

*Happy learning! 🎉*

---

**Quick Start Command:**
```bash
# Open all guides at once (macOS):
open NANO.md NANOMATH.md VERIFICATION.md

# Or just start reading:
cat NANO.md | less
```
