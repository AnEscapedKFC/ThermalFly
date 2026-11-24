"""OpenFly-Agent inference process. One JSON input line per policy decision."""

import argparse
import base64
import io
import json
import math
import sys
from collections import deque
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


# Vectors from OpenFly-Platform/train/eval.py::convert_to_action_id, in ID order.
ACTION_VECTORS = (
    (1, 0, 0, 0, 0, 0, 0, 0),
    (0, 3, 0, 0, 0, 0, 0, 0),
    (0, 0, 15, 0, 0, 0, 0, 0),
    (0, 0, 0, 15, 0, 0, 0, 0),
    (0, 0, 0, 0, 2, 0, 0, 0),
    (0, 0, 0, 0, 0, 2, 0, 0),
    (0, 0, 0, 0, 0, 0, 5, 0),
    (0, 0, 0, 0, 0, 0, 0, 5),
    (0, 6, 0, 0, 0, 0, 0, 0),
    (0, 9, 0, 0, 0, 0, 0, 0),
)


def decode_action(raw):
    if len(raw) != 8 or not all(math.isfinite(value) for value in raw):
        raise ValueError("OpenFly must produce eight finite action components")
    rounded = tuple(round(value) for value in raw)
    try:
        return ACTION_VECTORS.index(rounded)
    except ValueError:
        return None


def _unit_features(rows):
    import numpy as np

    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] == 0 or not np.isfinite(rows).all():
        raise ValueError("Expected a matrix of finite OpenFly features")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Empty OpenFly feature")
    return rows / norms


def _rbf_basis(distances_squared, width_squared):
    import numpy as np

    # Subtract the closest distance before exponentiating so queries between
    # demonstrations still have a well-defined local weighting.
    shifted = distances_squared - distances_squared.min(axis=1, keepdims=True)
    basis = np.exp(-0.5 * shifted / width_squared)
    return basis / basis.sum(axis=1, keepdims=True)


def fit_room_head(features, targets, path):
    """Fit a route-specific radial basis readout. Targets: forward/left/up m, yaw degrees, stop 0/1."""
    import numpy as np

    x = _unit_features(features)
    y = np.asarray(targets, dtype=np.float64)
    if len(x) < 2 or y.shape != (len(x), 5) or not np.isfinite(y).all():
        raise ValueError("Provide at least two features and matching finite five-column targets")
    if not np.isin(y[:, 4], (0, 1)).all():
        raise ValueError("STOP labels must be 0 or 1")
    center = x.mean(axis=0)
    support = x - center
    squared_norm = np.square(support).sum(axis=1)
    distances = np.maximum(squared_norm[:, None] + squared_norm - 2 * support @ support.T, 0)
    # Width follows the spacing between nearby training observations. Exclude
    # exact repetitions such as multiple identical terminal hover frames.
    neighbours = np.where(distances > 1e-8, distances, np.inf)
    np.fill_diagonal(neighbours, np.inf)
    nearest = np.sort(neighbours, axis=1)
    k = np.minimum(np.isfinite(nearest).sum(axis=1), 2) - 1
    if np.any(k < 0):
        raise ValueError("Need distinct visual-language observations to fit an indoor head")
    width_squared = float(np.median(nearest[np.arange(len(x)), k]))
    basis = _rbf_basis(distances, width_squared).astype(np.float64)
    gram = basis.T @ basis
    gram.flat[::len(gram) + 1] += 1e-4
    weights = np.linalg.solve(gram, basis.T @ y)
    with Path(path).open("wb") as dst:
        np.savez_compressed(dst, version=2, center=center, support=support, weights=weights.astype(np.float32),
                            targets=y.astype(np.float32), width_squared=width_squared)


class RoomHead:
    def __init__(self, path):
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            self.version = int(data["version"])
            if self.version not in (1, 2):
                raise ValueError("Unsupported indoor head version")
            self.center, self.support = data["center"], data["support"]
            self.weights, self.width_squared = data["weights"], float(data["width_squared"])
            self.targets = data["targets"] if self.version == 2 else None
        if (self.support.ndim != 2 or self.center.shape != self.support.shape[1:]
                or self.weights.shape != (len(self.support), 5) or not len(self.support)
                or not math.isfinite(self.width_squared) or self.width_squared <= 0
                or not all(np.isfinite(x).all() for x in (self.center, self.support, self.weights))):
            raise ValueError("Invalid indoor head dimensions or values")
        if self.targets is not None and (self.targets.shape != self.weights.shape or not np.isfinite(self.targets).all()):
            raise ValueError("Invalid indoor head target support")

    def predict(self, feature):
        import numpy as np

        query = _unit_features([feature])[0]
        if query.shape != self.center.shape:
            raise ValueError("Indoor head and OpenFly feature dimensions differ")
        distances = np.square(self.support - (query - self.center)).sum(axis=1)
        prediction = (_rbf_basis(distances[None], self.width_squared) @ self.weights)[0]
        if self.version == 2:
            # Far from demonstrations, ridge coefficients can extrapolate into
            # a turn unsupported by any nearby example. Blend into a convex
            # estimate from the nearest visual observations as support weakens.
            nearest = np.argsort(distances)[:4]
            local_targets = self.targets[nearest]
            weight = 1 / np.square(distances[nearest] + .02 * self.width_squared)
            local = np.average(local_targets, axis=0, weights=weight)
            trust = math.exp(-float(distances[nearest[0]]) / self.width_squared)
            bounded = np.clip(prediction, local_targets.min(axis=0), local_targets.max(axis=0))
            prediction = trust * bounded + (1 - trust) * local
        planar = np.linalg.norm(prediction[:2])
        if planar > 0.70:
            prediction[:2] *= 0.70 / planar
        prediction[2:] = np.clip(prediction[2:], [-0.30, -35, 0], [0.30, 35, 1])
        return {"waypoint": prediction[:4].tolist(), "stop": bool(prediction[4] >= .5),
                "stop_score": float(prediction[4])}


class OpenFlyPolicy:
    def __init__(self, args):
        train_dir = args.openfly_root.expanduser().resolve() / "train"
        if not (train_dir / "extern/hf/modeling_prismatic.py").is_file():
            raise FileNotFoundError(f"Missing OpenFly-Platform checkout under {train_dir.parent}")
        sys.path.insert(0, str(train_dir))

        import timm
        import torch
        from transformers import AutoTokenizer, BitsAndBytesConfig
        from extern.hf.configuration_prismatic import OpenFlyConfig
        from extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

        self.device = torch.device(args.device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA is unavailable for {args.device}")
            torch.cuda.set_device(self.device.index if self.device.index is not None else 0)
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        self.dtype = torch.bfloat16 if args.precision == "int4" else dtype[args.precision]
        self.unnorm_key = args.unnorm_key
        source = {"local_files_only": args.local_files_only}
        config = OpenFlyConfig.from_pretrained(args.checkpoint, **source)
        action_stats = (config.norm_stats or {}).get(self.unnorm_key, {}).get("action", {})
        if any(len(action_stats.get(key, [])) != 8 for key in ("q01", "q99")):
            raise ValueError(f"Checkpoint lacks eight-dimensional action statistics for {self.unnorm_key!r}")

        self.processor = PrismaticProcessor(
            image_processor=PrismaticImageProcessor.from_pretrained(args.checkpoint, **source),
            tokenizer=AutoTokenizer.from_pretrained(args.checkpoint, **source),
        )
        load_options = dict(config=config, torch_dtype=self.dtype, low_cpu_mem_usage=True,
                            device_map={"": str(self.device)}, attn_implementation="eager",
                            output_loading_info=True, **source)
        if args.precision == "int4":
            load_options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True,
            )

        # OpenFly's constructor asks TIMM to download both base vision models.
        # The published full checkpoint restores both fine-tuned backbones itself.
        create_model = timm.create_model

        def create_uninitialized_backbone(*model_args, **kwargs):
            kwargs["pretrained"] = False
            return create_model(*model_args, **kwargs)

        with patch.object(timm, "create_model", create_uninitialized_backbone):
            self.model, info = OpenVLAForActionPrediction.from_pretrained(args.checkpoint, **load_options)
        if info["missing_keys"] or info["mismatched_keys"]:
            raise ValueError(f"Incomplete checkpoint: missing={info['missing_keys']}, mismatched={info['mismatched_keys']}")
        self.model.eval()
        self.history = deque(maxlen=2)
        self.room_head = RoomHead(args.room_head) if getattr(args, "room_head", None) else None
        self.extract_features = getattr(args, "extract_features", False)
        self.visual_projection = None

    def fused_feature(self, inputs):
        import torch

        version = self.room_head.version if self.room_head is not None else 2
        # The last text token attends to the prepended current/history image
        # tokens. The projected camera patches are returned by the same pass.
        output = self.model(**inputs, output_hidden_states=True, output_projector_features=version == 2,
                            return_dict=True, use_cache=False)
        normalize = torch.nn.functional.normalize
        feature = normalize(output.hidden_states[-1][0, -1].float(), dim=0)
        if version == 2:
            # OpenFly compresses each history frame to one token, followed by
            # all 16x16 current RGB patches. Pool a 4x4 spatial layout and keep
            # a deterministic projection of the learned channels in each cell.
            projected = output.projector_features
            if projected is None or projected.shape[:2] != (1, 258):
                raise ValueError("Expected two history tokens and 256 current OpenFly patch tokens")
            current = normalize(projected[0, 2:].float(), dim=-1)
            pooled = current.reshape(4, 4, 4, 4, -1).mean(dim=(1, 3)).reshape(16, -1)
            if self.visual_projection is None:
                generator = torch.Generator(device="cpu").manual_seed(6502)
                projection = torch.randn((pooled.shape[-1], 96), generator=generator, dtype=torch.float32)
                self.visual_projection = projection.to(device=pooled.device)
            cells = normalize(pooled @ self.visual_projection, dim=-1)
            spatial = normalize(cells.flatten(), dim=0)
            feature = torch.cat((.5 * feature, math.sqrt(.75) * spatial))
        if not torch.isfinite(feature).all():
            raise ValueError("OpenFly returned a nonfinite fused prompt feature")
        return feature.cpu().numpy()

    def predict(self, instruction, image):
        import torch

        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Provide a nonempty navigation instruction")
        frames = [*self.history, image]
        frames = [frames[0]] * (3 - len(frames)) + frames
        inputs = self.processor(instruction, frames).to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            if self.extract_features or self.room_head is not None:
                feature = self.fused_feature(inputs)
                response = ({"feature": feature.tolist()} if self.extract_features
                            else self.room_head.predict(feature))
            else:
                # Upstream appends token 29871 only to IDs; generate can create
                # a matching mask for this unpadded single prompt. Generate all
                # eight tokens so its suffix decoder cannot include prompt IDs.
                inputs.pop("attention_mask", None)
                raw = self.model.predict_action(
                    **inputs, unnorm_key=self.unnorm_key, do_sample=False,
                    min_new_tokens=self.model.get_action_dim(self.unnorm_key),
                )
                values = [float(value) for value in raw]
                response = {"action_id": decode_action(values), "raw_action": values}
        self.history.append(image)
        return response


def serve(args, output):
    from PIL import Image

    policy = OpenFlyPolicy(args)
    ready = {"ready": True, "device": str(policy.device), "precision": args.precision,
             "unnorm_key": policy.unnorm_key}
    print(json.dumps(ready), file=output, flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("reset") is True:
                policy.history.clear()
                response = {"reset": True}
            else:
                data = base64.b64decode(request["image"], validate=True)
                with Image.open(io.BytesIO(data)) as frame:
                    response = policy.predict(request["instruction"], frame.convert("RGB"))
        except Exception as exc:
            response = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, allow_nan=False), file=output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openfly-root", required=True, type=Path)
    parser.add_argument("--checkpoint", default="IPEC-COMMUNITY/openfly-agent-7b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("int4", "bf16", "fp16", "fp32"), default="int4")
    parser.add_argument("--unnorm-key", default="vlnv1")
    parser.add_argument("--local-files-only", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--extract-features", action="store_true", help="Return fused prompt and current RGB spatial features.")
    mode.add_argument("--room-head", type=Path, help="Use a fitted indoor waypoint/STOP readout (.npz).")
    args = parser.parse_args()
    output = sys.stdout
    with redirect_stdout(sys.stderr):
        try:
            serve(args, output)
        except Exception as exc:
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=output, flush=True)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
