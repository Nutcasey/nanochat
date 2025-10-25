# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nanochat is a full-stack implementation of a ChatGPT-like LLM in a minimal, hackable codebase. It's designed to train models from scratch on a single 8XH100 GPU node, covering the entire pipeline: tokenization, pretraining, finetuning, evaluation, and web serving.

**Key Philosophy**: This is NOT a configurable framework with abstractions. It's a single, cohesive, minimal "strong baseline" designed to be readable, hackable, and forkable. Avoid adding configuration objects, factories, or complex abstractions.

## Development Setup

### Initial Setup
```bash
# Install uv package manager (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
uv sync
source .venv/bin/activate

# Install Rust for the tokenizer
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build the Rust BPE tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

### Running Tests
```bash
# Run tokenizer tests
python -m pytest tests/test_rustbpe.py -v -s

# Run all tests
python -m pytest tests/ -v -s
```

### Quick Training (Speedrun)
```bash
# Train the $100 tier model (~4 hours on 8XH100)
bash speedrun.sh

# Or in a screen session with logging
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# With wandb logging
WANDB_RUN=speedrun bash speedrun.sh
```

### Inference & Chat
```bash
# Chat via CLI
python -m scripts.chat_cli -p "Why is the sky blue?"

# Chat via web UI (ChatGPT-style interface)
python -m scripts.chat_web
```

## Architecture

### Core Model (nanochat/gpt.py)
The Transformer architecture uses modern features:
- **Rotary embeddings** (RoPE) instead of absolute positional embeddings
- **QK normalization** for attention stability
- **Untied embeddings**: separate weights for token embedding (`wte`) and language model head (`lm_head`)
- **ReLU² activation** in MLPs (instead of GeLU/SwiGLU)
- **RMSNorm** with no learnable parameters
- **No biases** in linear layers
- **Multi-Query Attention (MQA)** support for efficient inference

Model size is controlled by `depth` parameter:
- d20 (default speedrun): 561M params, 20 layers
- d26: ~1.2B params, 26 layers (GPT-2 grade)
- Model dimension = depth × 64 (aspect ratio of 64)
- Head dimension = 128

### Training Pipeline

The pipeline has 4 main phases:

1. **Tokenization** (`scripts/tok_train.py`, `scripts/tok_eval.py`)
   - Custom BPE tokenizer with vocab size 2^16 = 65,536
   - Trains on ~2B characters from FineWeb
   - Two implementations: RustBPE (training) and tiktoken (inference)
   - Special tokens for conversation formatting: `<|bos|>`, `<|user_start|>`, `<|assistant_start|>`, etc.

2. **Base Pretraining** (`scripts/base_train.py`, `scripts/base_eval.py`)
   - Trains on FineWeb dataset (250MB per shard)
   - Uses Chinchilla ratio: 20 tokens per parameter
   - Hybrid optimizer: **Muon** for weight matrices, **AdamW** for embeddings/lm_head
   - Evaluates on CORE metric (language understanding benchmark)
   - Data loaded via streaming from parquet files (`nanochat/dataloader.py`)

3. **Midtraining** (`scripts/mid_train.py`)
   - Teaches conversation special tokens, tool use, multiple choice format
   - Domain adaptation for chat-style interactions

4. **Supervised Finetuning (SFT)** (`scripts/chat_sft.py`)
   - Per-sequence domain adaptation
   - Trains on SmolTalk dataset
   - Evaluated on ARC-Challenge, ARC-Easy, GSM8K, HumanEval, MMLU, ChatCORE

5. **Reinforcement Learning (RL)** (`scripts/chat_rl.py`) *(optional)*
   - Currently only on GSM8K dataset
   - Comment in speedrun.sh to enable

### Inference Engine (nanochat/engine.py)

The `Engine` class provides efficient batched inference with:
- **KV caching** for autoregressive generation
- **Tool use** via calculator integration (`<|python_start|>` ... `<|python_end|>`)
- **Batch prefilling**: single prefill, then parallel decoding across multiple samples
- **Token forcing**: system can inject tokens (e.g., calculator outputs) during generation

Key feature: The engine maintains per-row state to track tool blocks and forced token queues.

### Data Pipeline

Data flows through:
1. `nanochat/dataset.py` - Downloads FineWeb shards from HuggingFace
2. `nanochat/dataloader.py` - Streams from parquet files, tokenizes on-the-fly, yields batches
3. Distributed data loading: each rank processes different shards (stride by world_size)

Environment variable `NANOCHAT_BASE_DIR` (default: `~/.cache/nanochat`) controls where data/checkpoints are stored.

### Evaluation Tasks (tasks/)

Benchmark suite includes:
- `arc.py` - ARC-Challenge and ARC-Easy (reasoning)
- `gsm8k.py` - Grade School Math 8K (math reasoning)
- `humaneval.py` - Code generation benchmark
- `mmlu.py` - Massive Multitask Language Understanding
- `smoltalk.py` - Chat evaluation (ChatCORE)
- `common.py` - CORE metric (base pretraining evaluation)

### Distributed Training

Uses PyTorch DDP (DistributedDataParallel):
```bash
# Single GPU (8x slower, but identical results via gradient accumulation)
python -m scripts.base_train

# Multi-GPU
torchrun --standalone --nproc_per_node=8 -m scripts.base_train
```

Key hyperparameters:
- `--device_batch_size`: Per-device batch size (reduce if OOM)
- `--total_batch_size`: Target global batch size in tokens (default: 524288)
- `--depth`: Model depth (20 for speedrun, 26 for GPT-2 grade)

The code automatically adjusts gradient accumulation steps to hit the target batch size.

## Key Implementation Details

### Optimizer Setup
- **Muon optimizer**: For all weight matrices in transformer blocks (higher efficiency for square matrices)
- **AdamW**: For embeddings and lm_head (non-square parameters)
- Learning rates scale by √(768/model_dim) for different model sizes

### Memory Management
- Embeddings use `bfloat16` to save memory
- Rotary embeddings are pre-computed and cached (10x sequence length)
- Gradients clipped at 1.0 by default
- Environment variable: `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`

### Checkpoint Management
Checkpoints stored in `$NANOCHAT_BASE_DIR/checkpoints/`:
- `base/` - After pretraining
- `mid/` - After midtraining
- `sft/` - After supervised finetuning
- `rl/` - After reinforcement learning (if run)

Each checkpoint includes:
- Model weights (`model.pt`)
- Optimizer states
- Metadata (step count, FLOPs, etc.)

### Report Generation
The `nanochat/report.py` module generates markdown reports:
```bash
python -m nanochat.report reset    # Clear and start new report
python -m nanochat.report generate # Generate final report.md
```

Reports include system info, metrics, evaluation results, and a summary table.

## Common Development Patterns

### Training a Larger Model
To scale up (e.g., d26 instead of d20):
1. Adjust data shards: params × 20 × 4.8 / 250M (Chinchilla calculation)
2. Reduce `--device_batch_size` if OOM (e.g., 32 → 16)
3. Pass `--depth=26` to training scripts

Example for d26:
```bash
python -m nanochat.dataset -n 450  # More data shards
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=26 --device_batch_size=16
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=16
```

### Modifying the Model Architecture
All architecture changes go in `nanochat/gpt.py`:
- `GPTConfig` dataclass defines hyperparameters
- Keep changes minimal and avoid abstractions
- Ensure changes work with both training and inference (Engine)

### Adding New Evaluation Tasks
1. Create new file in `tasks/` following existing patterns
2. Import and call from appropriate eval script (`scripts/base_eval.py` or `scripts/chat_eval.py`)
3. Update report generation if needed

### Working with the Tokenizer
Two tokenizer implementations:
- **RustBPE** (`rustbpe/`): High-performance Rust implementation for training
- **HuggingFace Tokenizer** (`nanochat/tokenizer.py`): Fallback Python implementation

After modifying Rust tokenizer:
```bash
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

## Wandb Integration

To enable wandb logging:
```bash
wandb login
WANDB_RUN=my_run_name bash speedrun.sh
```

Default is `WANDB_RUN=dummy` which disables wandb.

## File Organization

```
nanochat/              # Main package
├── gpt.py             # Transformer model architecture
├── engine.py          # Inference engine with KV cache
├── tokenizer.py       # Tokenizer implementations
├── dataloader.py      # Data streaming and tokenization
├── checkpoint_manager.py  # Checkpoint save/load
├── muon.py            # Muon optimizer implementation
├── adamw.py           # Distributed AdamW
├── report.py          # Report generation
├── common.py          # Shared utilities
└── ...

scripts/               # Training and inference scripts
├── tok_train.py       # Train tokenizer
├── base_train.py      # Pretrain base model
├── mid_train.py       # Midtraining
├── chat_sft.py        # Supervised finetuning
├── chat_rl.py         # Reinforcement learning
├── chat_web.py        # Web UI server
└── *_eval.py          # Evaluation scripts

tasks/                 # Evaluation benchmarks
rustbpe/              # Rust BPE tokenizer
speedrun.sh           # Main training pipeline script
run1000.sh            # $1000 tier training (not in master)
```

## Python Dependencies

Managed via `uv` and `pyproject.toml`. Key dependencies:
- PyTorch 2.8+ (with CUDA 12.8)
- HuggingFace datasets, tokenizers
- FastAPI + Uvicorn (for web UI)
- wandb (optional logging)
- tiktoken, regex (tokenization)

No heavyweight LLM frameworks (no Transformers library usage for the model itself).
