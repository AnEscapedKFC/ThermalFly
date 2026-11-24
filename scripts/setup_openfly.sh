#!/usr/bin/env bash
# Run with the simulator environment active. Model packages are installed in a venv.
set -eu
cache_dir="${OPENFLY_CACHE_DIR:-$HOME/.cache/openfly-room65}"
upstream_rev="c075075497a7122bad82f5b76b9be926ad5a81b3"
model_rev="21dcce235f1f2d40fc23c76998051abc5434cc99"
mkdir -p "$cache_dir"
checkout="$cache_dir/OpenFly-Platform"
if [ ! -e "$checkout" ]; then
    staging=$(mktemp -d "$cache_dir/openfly-source.XXXXXX")
    trap 'rm -rf -- "$staging"' EXIT
    git -C "$staging" init -q
    git -C "$staging" remote add origin https://github.com/SHAILAB-IPEC/OpenFly-Platform.git
    git -C "$staging" sparse-checkout init --cone
    git -C "$staging" sparse-checkout set train
    git -C "$staging" fetch --quiet --depth=1 --filter=blob:none origin "$upstream_rev"
    git -C "$staging" checkout -q --detach FETCH_HEAD
    mv -T "$staging" "$checkout"
fi
if [ "$(git -C "$checkout" rev-parse HEAD)" != "$upstream_rev" ]; then
    echo "Expected OpenFly source revision $upstream_rev in $checkout" >&2
    exit 1
fi
if [ ! -x "$cache_dir/venv/bin/python" ]; then
    python -m venv --system-site-packages "$cache_dir/venv"
fi
model_python="$cache_dir/venv/bin/python"
"$model_python" -m pip install transformers==4.48.1 tokenizers==0.21.1 timm==0.9.16 \
    accelerate==1.15.0 bitsandbytes==0.49.2 sentencepiece==0.2.3 safetensors==0.8.0 huggingface-hub==0.36.2
"$model_python" - "$cache_dir/openfly-agent-7b" "$model_rev" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("IPEC-COMMUNITY/openfly-agent-7b", local_dir=sys.argv[1], revision=sys.argv[2])
PY
