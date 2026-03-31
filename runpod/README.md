# RunPod (web terminal)

## 1. Clone upstream in the pod

```bash
cd /workspace   # or ~
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
pip install -r requirements.txt
```

## 2. Copy your changes in

From your laptop (small copy — **not** the whole repo), e.g. **RBL-ATN**:

| Copy into the clone | Role |
|---------------------|------|
| `train_gpt.py` | RBL hook + `TORCH_COMPILE` / `SDP_ALLOW_MATH` / single-job lock / defaults |
| `rbl_atn/` (entire folder) | Typed attention package |

**Ways to get them on the pod**

- **SCP** (two paths):  
  `scp -P <ssh_port> train_gpt.py root@<ip>:/workspace/parameter-golf/`  
  `scp -P <ssh_port> -r rbl_atn root@<ip>:/workspace/parameter-golf/`
- **Paste** file contents in the web terminal editor (`nano`/`vi`) if you only touched a few lines.
- **Your fork**: `git remote add mine <your-fork-url>` && `git fetch mine` && `git checkout mine/your-branch` (if you pushed there).

## 3. Shards — all from the web terminal

You **do not** SCP `.bin` data from your laptop. In the pod (same clone, from `parameter-golf/`):

```bash
export HF_TOKEN=hf_...   # optional; better rate limits on Hugging Face Hub
python data/cached_challenge_fineweb.py --train-shards 80 --variant sp1024
```

That pulls **`manifest.json`**, the **full val split**, the **tokenizer**, and **`--train-shards` many** `fineweb_train_*.bin` files into `data/datasets/…` / `data/tokenizers/`. Already-present files are **skipped** (`get()` is idempotent). Re-run the same command after a **new pod** or with a **larger `--train-shards`** when you want more train data.

## 4. Train (8× GPU on one node)

`train_gpt` needs **`WORLD_SIZE` to divide 8**. On 8 GPUs use **`torchrun`**:

```bash
cd /workspace/parameter-golf
export TRAIN_BATCH_TOKENS=524288
export ATTENTION_BACKEND=rbl
export TORCH_COMPILE=1
export MAX_WALLCLOCK_SECONDS=0
torchrun --standalone --nnodes=1 --nproc_per_node=8 --master_port=29500 train_gpt.py
```

Optional wrapper (same thing): `bash scripts/runpod_train.sh` (set `NPROC_PER_NODE=8`).

## Notes

- **RBL** default = **4 heads** (one per op); no GQA on that path.
- NCCL issues on some clouds: try `NCCL_P2P_DISABLE=1` and/or `NCCL_IB_DISABLE=1`.
- **`Dockerfile`** in repo root is optional if you prefer image-based pods; you don’t need it for clone + copy + run.
