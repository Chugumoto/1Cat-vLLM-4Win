import os
import sys

# nvidia ModelOpt weights are served via Xet; keep XET enabled.
os.environ.pop("HF_HUB_DISABLE_XET", None)

from huggingface_hub import snapshot_download

path = snapshot_download("nvidia/Qwen3.6-35B-A3B-NVFP4")
print(path, flush=True)
