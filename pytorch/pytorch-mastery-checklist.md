# PyTorch Mastery Checklist — From Python Dev to Large-Scale Training Pro

> **Goal:** By the end, you can implement *any* model in PyTorch and train it efficiently on a single GPU, multi-GPU, and multi-node clusters using the most scalable modern techniques (FSDP2, tensor/pipeline/context parallelism, FP8, `torch.compile`, distributed checkpointing).
>
> **How to use this:** Tick boxes as you go. Each **module** ends with a small, time-boxed 🧪 **assignment** to lock in the concepts before moving on. Assignments are deliberately tiny (mostly 20–90 min) — the point is *reps*, not big projects.
>
> **Legend:** `[ ]` = not done · `[x]` = done · ⭐ = a concept beginners usually skip but pros rely on.
>
> **Suggested pace:** Phase 1 in ~2 weeks, Phase 2 in ~2 weeks, Phase 3 in ~3–4 weeks, Phase 4 in ~1 week. Adjust freely.
>
> **Versions:** Written against **PyTorch 2.x** (2.6+). Distributed APIs move fast — the modern path is `fully_shard` (FSDP2) + `DTensor` + `DeviceMesh`. FSDP1 is deprecated (2.11+). When in doubt, check the official docs and the [`torchtitan`](https://github.com/pytorch/torchtitan) reference repo.

---

## 🧰 Module 0 — Setup & Mental Model

- [x] Install PyTorch with the correct CUDA build for your GPU (use the selector on pytorch.org)
- [x] Verify install: `torch.__version__`, `torch.cuda.is_available()`, `torch.cuda.get_device_name()`
- [x] Understand what PyTorch *is*: eager-mode tensor library + autograd + `nn` + distributed + a compiler (`torch.compile`)
- [x] Understand the ecosystem you'll touch: `torch`, `torchvision`/`torchaudio` (data), `torchao` (quant/FP8), `torchtitan` (reference LLM training), `torch.distributed` (scaling)
- [x] Know the difference between **eager mode** (default, Pythonic, debuggable) and **compiled/graph mode** (`torch.compile`, faster)
- [x] Set up a GPU environment you can actually run on (local GPU, Colab, or a rented cloud instance — you'll want ≥1 real GPU by Phase 2, and ≥2 GPUs by Phase 3)
- [x] ⭐ Learn to read the [official docs](https://docs.pytorch.org/docs/stable/) and [tutorials](https://docs.pytorch.org/tutorials/) — they are excellent and authoritative

> 🧪 **Assignment (10 min):** Write a 5-line script that prints your PyTorch version, whether CUDA is available, the GPU name, and creates one tensor on the GPU. Confirm it runs without error.

---

# PHASE 1 — Core PyTorch (single machine, single GPU)

## 🔢 Module 1 — Tensors

- [x] Create tensors: `torch.tensor`, `zeros`, `ones`, `arange`, `linspace`, `rand`, `randn`, `empty`, `full`, `eye`
- [x] `_like` variants (`zeros_like`, etc.) and why they're handy
- [x] **dtypes**: `float32`, `float16`, `bfloat16`, `int64`, `bool`, etc.; casting with `.to(dtype)` / `.float()`
- [x] **Devices**: `.to("cuda")`, `.cpu()`, `.to(device)`; the `device` object; setting a default device
- [x] Indexing & slicing; boolean/mask indexing; `torch.where`; fancy/advanced indexing; `gather`/`scatter`
- [x] Shape ops: `reshape`, `view`, `permute`, `transpose`, `squeeze`, `unsqueeze`, `flatten`, `expand`, `repeat`
- [x] ⭐ **`view` vs `reshape`**, what **contiguous** means, `.contiguous()`, and **strides**/storage (the mental model for *why* some ops are free and some copy)
- [x] **Broadcasting** rules (align trailing dims; size-1 dims stretch) — internalize this cold
- [x] Math ops & reductions: `sum`, `mean`, `max`/`min`, `argmax`, `norm`, `matmul`/`@`, `einsum`, and the `dim=`/`keepdim=` args
- [x] ⭐ **In-place ops** (trailing `_`, e.g. `add_`, `mul_`) — what they save and when they break autograd
- [x] Concatenation & splitting: `cat`, `stack`, `split`, `chunk`, `unbind`
- [x] NumPy interop: `.numpy()`, `torch.from_numpy`, and the shared-memory gotcha
- [x] ⭐ **CPU↔GPU transfer cost** and why you minimize `.item()`/`.cpu()` calls in hot loops

> 🧪 **Assignment (30 min):** Given a random tensor of shape `(32, 128)`, (1) normalize each row to unit L2 norm, (2) compute the pairwise cosine-similarity matrix `(32, 32)` using only vectorized ops (no Python loops), and (3) verify the diagonal is all ~1.0. Use `einsum` or `@` for the matmul.

---

## 🎯 Module 2 — Autograd (automatic differentiation)

- [ ] `requires_grad=True`; how PyTorch builds a **dynamic computational graph** on the fly
- [ ] `.backward()` and reading gradients from `.grad`
- [ ] Scalar vs non-scalar backward (the `gradient=` argument / why loss must be a scalar)
- [ ] ⭐ **`torch.no_grad()` vs `torch.inference_mode()`** — when to use each (inference_mode is stricter/faster)
- [ ] `.detach()` — cut a tensor out of the graph; when you need it
- [ ] **Gradient accumulation**: grads *sum* across `.backward()` calls until zeroed — the basis for large effective batch sizes
- [ ] Zeroing grads: `optimizer.zero_grad()` and ⭐ `set_to_none=True` (default now; slightly faster)
- [ ] `retain_graph`, `create_graph` (higher-order gradients)
- [ ] ⭐ Writing a **custom `torch.autograd.Function`** (custom forward + backward) — you'll need this for custom kernels/ops
- [ ] Debugging: `torch.autograd.set_detect_anomaly(True)` to find NaNs/bad backward

> 🧪 **Assignment (25 min):** Implement `y = sin(x²)` for a scalar `x`, call `.backward()`, and confirm PyTorch's `x.grad` equals the hand-derived derivative `2x·cos(x²)` at three different `x` values. Then do it *without* autograd using `torch.autograd.grad`, and confirm they match.

---

## 🧱 Module 3 — Building Models with `nn.Module`

- [ ] `nn.Module`: `__init__` + `forward`; why you never call `.forward()` directly (call `model(x)`)
- [ ] **Parameters vs buffers**: `nn.Parameter`, `register_buffer` (e.g., running stats, positional encodings)
- [ ] Core layers: `Linear`, `Embedding`, `Conv2d`, `LayerNorm`/`RMSNorm`, `Dropout`, `BatchNorm`
- [ ] Activations: `ReLU`, `GELU`, `SiLU`/`Swish`, `Softmax`, and functional vs module forms (`F.relu` vs `nn.ReLU`)
- [ ] Containers: `nn.Sequential`, `nn.ModuleList`, `nn.ModuleDict` (and why a plain Python list of layers *won't* register params)
- [ ] Inspecting models: `.parameters()`, `.named_parameters()`, `.modules()`, `.state_dict()`, counting params
- [ ] ⭐ **Weight initialization**: `nn.init`, why init matters, custom `apply(init_fn)`
- [ ] `model.train()` vs `model.eval()` and what actually changes (Dropout, BatchNorm)
- [ ] **Save/load**: `state_dict` + `torch.save`/`load` (save the state_dict, *not* the whole model); `load_state_dict(strict=...)`
- [ ] ⭐ `.to(device)` / `.half()` / `.bfloat16()` on a whole module; moving models correctly
- [ ] ⭐ **Meta device** init (`device="meta"`) — build huge models with no memory, then materialize (essential at scale)

> 🧪 **Assignment (40 min):** Build a small MLP (`2 → 16 → 16 → 1`) as an `nn.Module` and train it to solve XOR (4 data points). Confirm it reaches ~0 loss. Then save the `state_dict`, reload it into a fresh model instance, and verify identical outputs.

---

## 📉 Module 4 — Losses, Optimizers & Schedulers

- [ ] Loss functions: `CrossEntropyLoss` (⭐ expects raw logits + class indices, has softmax built in), `MSELoss`, `BCEWithLogitsLoss`, `NLLLoss`
- [ ] ⭐ Understand `reduction=` (`mean`/`sum`/`none`) and why it matters for grad accumulation & masking
- [ ] Optimizers: `SGD` (+ momentum), `Adam`, **`AdamW`** (the LLM default), `RMSprop`; the `optimizer.step()` loop
- [ ] ⭐ **Parameter groups**: different LR/weight-decay per group (e.g., no weight decay on norms/biases)
- [ ] `optimizer.state_dict()` — optimizer state is huge for Adam (2× params); you must checkpoint it
- [ ] LR schedulers: `StepLR`, `CosineAnnealingLR`, `OneCycleLR`, `LambdaLR`; **warmup + cosine decay** (LLM standard)
- [ ] ⭐ **Gradient clipping**: `torch.nn.utils.clip_grad_norm_` — prevents blow-ups in transformers
- [ ] `foreach` / `fused` optimizer implementations (⭐ `fused=True` for speed on GPU)

> 🧪 **Assignment (30 min):** Take your XOR MLP. Switch the optimizer to `AdamW` with two param groups (weight decay on weights, none on biases). Add a warmup-then-cosine LR schedule over 200 steps and log the LR each step. Plot or print the LR curve to confirm the warmup + decay shape.

---

## 📦 Module 5 — Data Loading

- [ ] `Dataset` (map-style: `__len__` + `__getitem__`) — write a custom one
- [ ] `DataLoader`: `batch_size`, `shuffle`, `drop_last`
- [ ] ⭐ **`num_workers`** (parallel loading), **`pin_memory=True`**, and **`non_blocking=True`** transfers — the standard fast-input-pipeline combo
- [ ] `prefetch_factor`, `persistent_workers` for throughput
- [ ] **`collate_fn`**: custom batching, padding variable-length sequences
- [ ] `IterableDataset` for streaming / sharded / infinite datasets (what you use for large corpora)
- [ ] Samplers: `RandomSampler`, `WeightedRandomSampler`, and a preview of `DistributedSampler` (Module 13)
- [ ] ⭐ Diagnosing an **input-bound** pipeline (GPU starving while CPU loads) vs compute-bound

> 🧪 **Assignment (35 min):** Write a custom `Dataset` that yields `(sequence, length)` pairs of *random* integer sequences with lengths 5–15. Write a `collate_fn` that pads a batch to the max length in that batch and returns a padding mask. Iterate one batch with `num_workers=2` and print the padded shape + mask.

---

## 🔁 Module 6 — The Training Loop & Reproducibility

- [ ] Write the canonical loop from scratch: `for epoch → for batch → forward → loss → zero_grad → backward → clip → step → (scheduler.step)`
- [ ] Separate **train vs eval loops** (`model.eval()` + `inference_mode()` for validation)
- [ ] Tracking metrics correctly (accumulate `loss.item()` weighted by batch size; avoid syncing every step)
- [ ] **Checkpointing**: save model + optimizer + scheduler + epoch/step + RNG state; resume cleanly
- [ ] ⭐ **Reproducibility**: `torch.manual_seed`, `torch.cuda.manual_seed_all`, `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic`/`benchmark`, and worker seeding
- [ ] Logging: TensorBoard (`torch.utils.tensorboard`) or Weights & Biases
- [ ] Early stopping / best-checkpoint tracking
- [ ] ⭐ Sanity checks every pro runs: **overfit a single batch** to ~0 loss (proves the model+loss+loop are wired correctly)

> 🧪 **Assignment (45 min):** Train a small CNN or MLP on a tiny dataset (e.g., a 500-sample subset of MNIST/FashionMNIST or even synthetic data) for a few epochs with train/val split, checkpointing the best model. Then **deliberately overfit a single batch** to near-zero loss as a wiring sanity check. Reload the best checkpoint and report val accuracy.

---

# PHASE 2 — Single-GPU Performance & Modern Building Blocks

## ⚙️ Module 7 — GPU Mechanics & Memory

- [ ] CUDA semantics: ops are **asynchronous**; the Python call returns before the GPU finishes
- [ ] ⭐ **Synchronization**: `torch.cuda.synchronize()` — required for *correct timing* (else you time the launch, not the work)
- [ ] CUDA **streams** and events (basic awareness; overlap of compute/copy)
- [ ] The **caching allocator**: why `nvidia-smi` shows memory that "isn't freed"; `torch.cuda.empty_cache()`
- [ ] Memory accounting: `memory_allocated`, `memory_reserved`, `max_memory_allocated`, `reset_peak_memory_stats`
- [ ] ⭐ **Debugging OOM**: reduce batch, grad accumulation, activation checkpointing, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- [ ] ⭐ **Memory snapshot** tool (`torch.cuda.memory._record_memory_history` + visualizer) to *see* what's using memory
- [ ] Understand the four memory consumers: **parameters, gradients, optimizer states, activations** (this framing drives every scaling decision)

> 🧪 **Assignment (30 min):** Take a model from Phase 1. Correctly time one forward+backward pass using `torch.cuda.synchronize()` before/after (compare against the naive, wrong timing without sync). Then print peak memory with `max_memory_allocated()`, double the batch size, and report how peak memory changes.

---

## 🎚️ Module 8 — Mixed Precision & Numerics

- [ ] Why lower precision: less memory + faster tensor-core matmuls
- [ ] `float16` vs ⭐ **`bfloat16`** (wider range, no loss scaling needed — the modern default on Ampere+)
- [ ] **AMP**: `torch.autocast(device_type="cuda", dtype=...)` context manager
- [ ] ⭐ **`GradScaler`** — needed for `float16` (prevents underflow); usually *not* needed for `bfloat16`
- [ ] The correct AMP training-loop pattern (autocast forward+loss; scale/step for fp16)
- [ ] ⭐ `torch.set_float32_matmul_precision("high")` / TF32 — free speedup on matmuls
- [ ] Which ops stay in fp32 (reductions, softmax, norms) and why — numerical stability
- [ ] Preview: **FP8** training exists too (Module 15) via `torchao`

> 🧪 **Assignment (30 min):** Retrain a Phase-1 model twice — once in fp32, once with `bfloat16` autocast. Compare peak memory and per-step time. Confirm final loss is comparable. Then try `float16` *with* a `GradScaler` and confirm it also trains.

---

## 🔬 Module 9 — Profiling & Bottleneck Analysis

- [ ] ⭐ **`torch.profiler`**: profile CPU + CUDA, export a Chrome/Perfetto trace, read the timeline
- [ ] `profiler.key_averages()` tables; finding the top time-consuming ops
- [ ] Identify the bottleneck class: **compute-bound vs memory-bandwidth-bound vs input/data-bound vs CPU-launch-bound (kernel-launch overhead)**
- [ ] Spotting **CPU↔GPU sync points** that serialize execution (e.g., `.item()`, `.cpu()`, `print(tensor)`, Python control flow on tensor values)
- [ ] `torch.utils.benchmark.Timer` for micro-benchmarking ops fairly
- [ ] ⭐ Reading GPU utilization / occupancy at a high level; knowing when you're leaving perf on the table

> 🧪 **Assignment (40 min):** Profile one training step of a Phase-1 model with `torch.profiler`. Export the trace, open it (Perfetto/TensorBoard), and identify the single most expensive operation. Write one sentence naming the bottleneck class (compute / memory / data / launch overhead) and your evidence.

---

## 🚀 Module 10 — `torch.compile` & Kernel-Level Speed

- [ ] `torch.compile(model)` basics; it's a JIT — the **first step is slow** (compile), later steps are fast
- [ ] Compile **modes**: `default`, `reduce-overhead`, `max-autotune` — the tradeoffs
- [ ] The stack conceptually: **TorchDynamo** (capture) → **TorchInductor** (codegen, often Triton kernels)
- [ ] ⭐ **Graph breaks**: what causes them (unsupported ops, data-dependent Python control flow) and why they kill speedups
- [ ] `fullgraph=True` to *force* a single graph and surface breaks; fixing breaks
- [ ] ⭐ **Recompilation** from dynamic shapes; `dynamic=True`, marking dynamic dims, and **bucketing/padding** variable shapes
- [ ] ⭐ Debugging with `TORCH_LOGS="graph_breaks,recompiles,perf_hints"`
- [ ] ⭐ **Regional compilation**: compile the *repeated block* (e.g., one transformer layer) instead of the whole model — much faster warmup, near-full speedup, standard for LLMs
- [ ] **CUDA Graphs** (`reduce-overhead` mode / CUDA Graph Trees) — eliminate per-kernel launch overhead for small ops
- [ ] Composability note: `torch.compile` works *with* AMP, FSDP2, and `torchao` (you'll stack these)

> 🧪 **Assignment (45 min):** Take a model with a repeated block (stack 6 identical Linear+GELU blocks). Benchmark eager vs `torch.compile` (full model) vs **regional** compilation of one block. Report warmup time and steady-state per-step time for each. Turn on `TORCH_LOGS="graph_breaks,recompiles"` and confirm whether any graph breaks occur.

---

## 🧠 Module 11 — Transformers From Scratch + Efficient Attention + Checkpointing

- [ ] Implement **multi-head self-attention** from scratch (Q/K/V projections, scaled dot-product, causal mask, output projection)
- [ ] ⭐ **`F.scaled_dot_product_attention` (SDPA)** — the fused, memory-efficient attention (FlashAttention under the hood); its backends (flash / mem-efficient / math) and how to select/inspect them
- [ ] Build a full **transformer block**: attention + MLP + residuals + norm (pre-norm), then stack into a model
- [ ] Positional encodings: learned, sinusoidal, and ⭐ **RoPE** (rotary) — the modern LLM choice
- [ ] Token **embedding** + LM head + weight tying
- [ ] ⭐ **Activation (gradient) checkpointing** (`torch.utils.checkpoint`) — recompute activations in backward to trade compute for memory (essential for large models)
- [ ] KV-cache concept for inference (Module 16 goes deeper)
- [ ] ⭐ Put it together: a tiny **GPT-style** model you fully understand end-to-end

> 🧪 **Assignment (75 min):** Implement a minimal GPT block (causal multi-head attention via SDPA + MLP + residual + norm), stack 2–4 of them into a tiny LM (small vocab, short context), and train it to memorize a short text string (overfit). Then wrap the blocks in activation checkpointing and confirm training still works while peak activation memory drops.

---

# PHASE 3 — Distributed & Large-Scale Training

> You need ≥2 GPUs for most of this. If you only have 1 GPU, you can still learn the **APIs** by launching multiple processes (even the `gloo` CPU backend works for collectives practice), and read [`torchtitan`](https://github.com/pytorch/torchtitan) as the canonical example. Real speedups require real multi-GPU/multi-node hardware.

## 🌐 Module 12 — Distributed Foundations

- [ ] The parallelism taxonomy (know what each *is* and *when* to use it):
  - **Data Parallel (DDP)** — replicate model, split data
  - **Sharded Data Parallel (FSDP2 / ZeRO)** — shard params/grads/optimizer states
  - **Tensor Parallel (TP)** — split individual weight matrices across GPUs
  - **Pipeline Parallel (PP)** — split layers into stages across GPUs
  - **Context/Sequence Parallel (CP)** — split the sequence dimension (long context)
  - **Expert Parallel (EP)** — split MoE experts across GPUs
- [ ] `torch.distributed`: **process group**, **backend** (⭐ `nccl` for GPU, `gloo` for CPU), init
- [ ] The trio you must know cold: ⭐ **`world_size`**, **`rank`** (global), **`local_rank`** (within a node)
- [ ] Environment variables: `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`, `LOCAL_RANK`
- [ ] ⭐ **Collective operations**: `all_reduce`, `all_gather`, `reduce_scatter`, `broadcast`, `reduce`, `barrier` — and *conceptually* what each does to data across ranks
- [ ] Point-to-point: `send`/`recv` (used by pipeline parallel)
- [ ] ⭐ **`torchrun`** (elastic launcher): how it sets env vars and spawns one process per GPU
- [ ] ⭐ **`DeviceMesh`** (`init_device_mesh`) — the modern abstraction that names your GPU grid (e.g., a 2D mesh of `["dp", "tp"]`); *everything* modern (FSDP2/TP/PP/CP) composes on top of it
- [ ] Correctness habits: seed per rank appropriately, guard rank-0-only logging/saving, always `barrier`/cleanup

> 🧪 **Assignment (45 min):** Write a script launched with `torchrun --nproc_per_node=2` (2 GPUs, or 2 CPU procs with `gloo`) where each rank creates a tensor equal to its rank, then performs an `all_reduce(SUM)` and prints the result. Confirm every rank prints the same summed value. Then build a `2×1` `DeviceMesh` and print each rank's coordinates.

---

## 🔀 Module 13 — Data Parallelism (DDP & HSDP)

- [ ] ⭐ **`DistributedDataParallel` (DDP)**: wrap the model; each rank has a full replica and a data shard
- [ ] How DDP works: local backward → ⭐ **gradient all-reduce** (bucketed, overlapped with backward) → identical grads everywhere → identical step
- [ ] **`DistributedSampler`** + ⭐ **`sampler.set_epoch(epoch)`** every epoch (or your shuffling breaks across epochs)
- [ ] ⭐ **`no_sync()`** context for gradient accumulation (skip the all-reduce until the last micro-step)
- [ ] `find_unused_parameters`, `static_graph=True`, gradient bucketing knobs
- [ ] ⭐ **Effective batch size** = per-GPU batch × world size × grad-accum steps; LR scaling considerations
- [ ] Know that **`nn.DataParallel` is legacy** — do **not** use it; DDP (or FSDP2) is the answer
- [ ] **HSDP (Hybrid Sharded Data Parallel)**: shard *within* a node, replicate *across* nodes (2D mesh) — best of both when scaling out
- [ ] Saving/loading with DDP (unwrap `.module`, save on rank 0)

> 🧪 **Assignment (50 min):** Convert your Module 6 training script to **DDP** with `torchrun` on 2 processes. Add a `DistributedSampler` with `set_epoch`. Verify (a) both ranks converge and (b) the loss curve roughly matches your single-GPU run at the equivalent *effective* batch size. Bonus: add `no_sync()` and do 2 grad-accum steps.

---

## 🧩 Module 14 — Sharding & Model Parallelism (the real scaling)

### FSDP2 — the workhorse
- [ ] ⭐ **FSDP2 = ZeRO-style sharding**: params, grads, **and** optimizer states are sharded across ranks (breaks the "model must fit on one GPU" barrier)
- [ ] The mechanism: fully sharded at rest → **all-gather** params before fwd/bwd → **reduce-scatter** grads in backward → optimizer updates *sharded* params
- [ ] ⭐ The modern API: **`fully_shard`** (FSDP2), *not* the deprecated `FullyShardedDataParallel` wrapper
- [ ] ⭐ **Apply bottom-up**: shard each transformer block first, then the root model
- [ ] Sharding strategies: `FULL_SHARD`, `HYBRID_SHARD` (= HSDP), `NO_SHARD`
- [ ] `MixedPrecisionPolicy` (param/reduce/output dtypes), CPU offload, and prefetch controls (`set_modules_to_forward/backward_prefetch`)
- [ ] ⭐ **Meta-device init** + FSDP2 to build models too big for one GPU, then materialize shards
- [ ] ⭐ Build the optimizer **after** sharding (it must see the sharded/DTensor params)
- [ ] Why FSDP2 > FSDP1: per-parameter **DTensor** sharding, communication-free sharded state dicts, deterministic memory, composability, FP8 support

### DTensor — the foundation
- [ ] ⭐ **`DTensor`**: a "global tensor" whose data is sharded over a `DeviceMesh`
- [ ] **Placements**: `Shard(dim)`, `Replicate()`, `Partial()`; `distribute_tensor`, `redistribute`
- [ ] How DTensor desugars into collectives (and is compilable by `torch.compile`)

### Tensor / Sequence Parallel (TP/SP)
- [ ] ⭐ TP splits weight matrices: **`ColwiseParallel`** and **`RowwiseParallel`** (column-split then row-split composes into one all-reduce per block)
- [ ] `parallelize_module(model, mesh, plan)` with a TP plan
- [ ] ⭐ **Sequence Parallel** for the norm/dropout regions; ⭐ **loss parallel** (shard the vocab logits to save the huge softmax memory)
- [ ] When TP helps: very large layers; keep TP *within* a node (needs high intra-node bandwidth / NVLink)

### Pipeline Parallel (PP)
- [ ] ⭐ Split the model into **stages** by layer; pass activations stage→stage
- [ ] **Micro-batching** and the **pipeline bubble**; schedules: **GPipe**, **1F1B**, **interleaved 1F1B**
- [ ] PyTorch pipelining APIs (`torch.distributed.pipelining`): `pipeline`, `PipelineStage`, schedule classes

### Context Parallel (CP) & Expert Parallel (EP)
- [ ] ⭐ **Context Parallel**: shard the **sequence** across GPUs via **Ring Attention** — enables very long context
- [ ] **Expert Parallel** for **MoE**: route tokens to experts living on different GPUs (all-to-all comms)

### Composition — nD parallelism
- [ ] ⭐ **Compose them on one `DeviceMesh`**: e.g., 2D (FSDP×TP), up to **4D** (DP/FSDP × TP × PP × CP) — this is how trillion-param models are trained
- [ ] ⭐ Study **[`torchtitan`](https://github.com/pytorch/torchtitan)** — the PyTorch-native reference that composes FSDP2 + TP + PP + CP + FP8 + `torch.compile` + distributed checkpointing for Llama-class models
- [ ] Rule of thumb ordering: TP within a node (NVLink) → PP/DP across nodes → FSDP for memory → CP for long sequences

> 🧪 **Assignment (75–90 min):** On 2+ GPUs, take your tiny GPT from Module 11 and shard it with **FSDP2 `fully_shard`** (apply per-block bottom-up, then root; build optimizer after). Confirm it trains and that **per-GPU memory is lower** than the DDP version. Then read the `torchtitan` `parallelize.py` for Llama and write 3 bullet points explaining how it applies FSDP2 + TP on a DeviceMesh. *(If you only have 1 GPU: skip the FSDP run, do the reading + write a TP plan on paper for a single transformer block using Colwise/Rowwise.)*

---

## 📈 Module 15 — Large-Scale Training Efficiency

- [ ] ⭐ **FP8 training** with **`torchao.float8`**: scaling strategies (tensorwise/rowwise; dynamic/delayed/static), composes with FSDP2 (**Float8 all-gather**) + `torch.compile`; know *when* FP8 matmul beats bf16 (large M/K/N)
- [ ] ⭐ **Communication/computation overlap**: the key to scaling efficiency (FSDP2 prefetch, DDP bucket overlap, async TP) — and how to verify it in a profile
- [ ] ⭐ **Distributed Checkpointing (DCP)**: `torch.distributed.checkpoint` — sharded, parallel, resumable checkpoints; ⭐ **async checkpointing** (overlaps saving with training; 5–15× less overhead)
- [ ] `get_model_state_dict` / `set_model_state_dict` (with `broadcast_from_rank0`) — the DTensor-aware way (naïve `torch.save` won't cut it at scale)
- [ ] ⭐ **Fault tolerance & elasticity**: elastic `torchrun`, **`torchft`**, and ⭐ **Flight Recorder** for diagnosing hung/crashed collective jobs
- [ ] ⭐ **Throughput metrics**: **MFU** (Model FLOPs Utilization), tokens/sec, samples/sec, and measuring **scaling efficiency** (does 2× GPUs give ~2× throughput?)
- [ ] Gradient accumulation at scale + effective global batch tuning
- [ ] Efficient large-scale data pipelines (sharded/streaming datasets, avoiding the input bottleneck at N GPUs)
- [ ] ⭐ Selective activation checkpointing (checkpoint only expensive layers) to balance the memory/compute tradeoff
- [ ] Awareness: NCCL tuning, network topology (NVLink vs InfiniBand), and where comms become the bottleneck

> 🧪 **Assignment (60 min):** On your FSDP2 GPT, (1) add `torch.compile` (regional, per block) and measure the throughput change; (2) compute a rough **tokens/sec** and estimate **MFU** for your GPU; (3) save a checkpoint with **DCP** and resume from it, confirming loss continues smoothly. *(1-GPU fallback: do the compile + tokens/sec + MFU estimate on a single-GPU model, and save/resume with `get/set_model_state_dict`.)*

---

# PHASE 4 — Inference, Quantization & Serving

## 🔧 Module 16 — Quantization, Export & Serving

- [ ] Inference basics done right: `model.eval()` + `inference_mode()`; no grad, no optimizer state
- [ ] ⭐ **KV cache**: cache past K/V so generation is O(n) not O(n²); how it changes the memory profile
- [ ] Batched/streaming generation; greedy vs sampling (temperature/top-k/top-p) at a high level
- [ ] ⭐ **Quantization** with **`torchao`**: weight-only vs dynamic; **Int8**, **Int4**, **FP8** (and emerging MX/NVFP4 formats)
- [ ] **PTQ** (post-training quantization) vs ⭐ **QAT** (quantization-aware training — recovers accuracy)
- [ ] ⭐ `torchao` composes with `torch.compile` + FSDP2, and integrates with HuggingFace + vLLM out of the box
- [ ] **Export paths**: ⭐ `torch.export` → **AOTInductor** (ahead-of-time compiled artifact for C++/server deployment); **ExecuTorch** for on-device/mobile
- [ ] ⭐ **Serving frameworks**: **vLLM** and **SGLang** (paged attention, continuous batching, high-throughput LLM serving); know that TorchServe is legacy
- [ ] Measuring serving perf: throughput (tokens/sec), latency (TTFT / inter-token), and the throughput↔latency tradeoff; FP8 serving throughput/latency wins
- [ ] ⭐ Continuous (in-flight) batching & PagedAttention *concepts* — why they beat naive batching

> 🧪 **Assignment (60 min):** Take your tiny trained GPT (or a small HF model). (1) Implement/enable a **KV cache** and measure generation speedup vs no-cache. (2) Quantize it to Int8 or Int4 with `torchao` and compare model size + output quality on a couple of prompts. *(Optional stretch: serve a small quantized HF model with vLLM and hit it with one request.)*

---

# 🏆 Capstone Project — Train & Serve a Small LLM at Scale

Bring it all together into one project. Keep the model small (e.g., a ~10–50M param GPT) so iterations are fast — the point is exercising the *full pipeline*.

- [ ] **Model**: implement a GPT-style transformer from scratch (RoPE, SDPA/flash attention, RMSNorm, weight tying)
- [ ] **Data**: a streaming/sharded `IterableDataset` over a real text corpus (e.g., a small slice of a public dataset) with a fast `DataLoader` (workers + pinned memory)
- [ ] **Single-GPU baseline**: bf16 autocast + `torch.compile` (regional) + activation checkpointing; overfit-a-batch sanity check first
- [ ] **Scale it**: shard with **FSDP2** on 2+ GPUs; add **TP** on a 2D `DeviceMesh` if you have ≥4 GPUs
- [ ] **Efficiency**: enable **FP8** via `torchao` (if hardware supports it), verify **comms/compute overlap** in a profiler trace, and report **tokens/sec + MFU**
- [ ] **Robustness**: **DCP** checkpointing with resume; confirm you can kill and restart the run cleanly
- [ ] **Correctness**: loss curve is sane; validation perplexity decreases; multi-GPU matches single-GPU at equal effective batch
- [ ] **Serve it**: quantize with `torchao` (Int4/Int8/FP8) and run inference with a KV cache; optionally deploy via vLLM or export via AOTInductor
- [ ] **Write-up**: a short README documenting your parallelism config, the four memory consumers' footprint, throughput numbers, and what you'd change to scale to 100s of GPUs

> ✅ **You're done when:** you can start from an empty file and, without copying, stand up a transformer, train it efficiently on multiple GPUs with a sensible parallelism strategy, checkpoint/resume it, and serve a quantized version — while being able to explain *why* each scaling choice was made.

---

## 📚 Key References (bookmark these)

- **Official docs & tutorials** — `docs.pytorch.org/docs/stable` and `docs.pytorch.org/tutorials` (start here for anything)
- **`torchtitan`** — `github.com/pytorch/torchtitan` — the canonical PyTorch-native large-scale LLM training reference (FSDP2 + TP + PP + CP + FP8 + compile + DCP)
- **`torchao`** — `github.com/pytorch/ao` — quantization + FP8 for training and inference
- **FSDP2 tutorial** — the "Getting Started with Fully Sharded Data Parallel (FSDP2)" tutorial
- **`torch.distributed`** overview + DeviceMesh / DTensor / TP / Pipelining tutorials
- **vLLM docs** — for serving

---

*Tip: Don't just tick boxes — for each ⭐ item, be able to answer "what problem does this solve and what breaks without it?" That's the difference between using PyTorch and mastering it.*
