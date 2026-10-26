import argparse
import csv
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def marker_catalog(state, checkpoint, catalog=None):
    if "input_marker_ids" in state:
        return np.asarray(state["input_marker_ids"], dtype=str)
    if catalog is None:
        for folder in Path(checkpoint).resolve().parents:
            candidate = folder / "marker_ids.npy"
            if candidate.is_file():
                catalog = candidate
                break
    if catalog is None:
        raise ValueError("The original checkpoint requires --marker-catalog marker_ids.npy")
    return np.asarray(np.load(catalog, allow_pickle=False), dtype=str)


def align_markers(state, raw, supplied_ids, original_ids):
    raw = np.asarray(raw)
    supplied_ids = np.asarray(supplied_ids, dtype=str)
    original_ids = np.asarray(original_ids, dtype=str)
    cols = np.asarray(state["marker_cols"], dtype=np.int64)
    if raw.ndim != 2 or supplied_ids.ndim != 1 or raw.shape[1] != len(supplied_ids):
        raise ValueError("dosage columns must match the one-dimensional marker_id array")
    if len(np.unique(supplied_ids)) != len(supplied_ids):
        raise ValueError("Duplicate input marker IDs")
    if original_ids.ndim != 1 or len(np.unique(original_ids)) != len(original_ids):
        raise ValueError("The checkpoint marker catalog must contain unique IDs")
    if cols.ndim != 1 or len(cols) == 0 or cols.min() < 0 or cols.max() >= len(original_ids):
        raise ValueError("Invalid checkpoint marker positions")
    if np.array_equal(supplied_ids, original_ids):
        return raw[:, cols]
    lookup = {name: i for i, name in enumerate(supplied_ids)}
    needed = original_ids[cols]
    missing = [name for name in needed if name not in lookup]
    if missing:
        raise ValueError(f"Missing {len(missing)} required markers, including {missing[:5]}")
    return raw[:, [lookup[name] for name in needed]]


def predict_values(state, raw_qc):
    """raw_qc contains dosage columns in the checkpoint's QC order."""
    x = np.asarray(raw_qc, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(state["marker_cols"]):
        raise ValueError("Incorrect QC genotype shape")
    if not np.isin(x, [-1, 0, 1, 2]).all():
        raise ValueError("Use 0/1/2 alternate-allele dosage and -1 for missing calls")
    x = np.where(x < 0, state["imputation_mean"], x)
    xb = ((x[:, state["b_keep"]] - state["b_mu"]) / state["b_sd"]).astype(np.float32)
    xd = ((x[:, state["d_pos"]] - state["d_mu"]) / state["d_sd"]).astype(np.float32)
    weights = state["marker_weights"]
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Invalid checkpoint marker weights")
    if not np.isfinite(state["sd"]) or float(state["sd"]) <= 0:
        raise ValueError("Invalid training phenotype scale")
    kd = (xd * (weights / weights.sum())[None]) @ state["lib_deep"].T
    standardized = (float(state["standardized_intercept"]) + xb @ state["base_coefficient"]
                    + float(state["beta"]) * (kd @ state["alpha"]))
    prediction = standardized.astype(np.float64) * float(state["sd"]) + float(state["mean"])
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Non-finite prediction")
    return prediction


def predict_checkpoint(checkpoint, raw, marker_ids, catalog=None):
    if Path(checkpoint).suffix.lower() in {'.pt', '.pth'}:
        from .model import load_full_checkpoint
        net, transform, _, _ = load_full_checkpoint(checkpoint, marker_catalog=catalog)
        return predict_full_values(net, transform, raw, marker_ids)
    state = load_checkpoint(checkpoint)
    order = marker_catalog(state, checkpoint, catalog)
    aligned = align_markers(state, raw, marker_ids, order)
    return predict_values(state, aligned)


def predict_full_values(net, transform, raw, marker_ids):
    import torch
    aligned = align_markers({'marker_cols': transform.marker_cols}, raw, marker_ids,
                            transform.input_marker_ids)
    if not np.isin(aligned, [-1, 0, 1, 2]).all():
        raise ValueError('Use 0/1/2 alternate-allele dosage and -1 for missing calls')
    imputed = np.where(aligned < 0, 2 * transform.af, aligned).astype(np.float64)
    encoded = transform.encode_imputed(imputed)
    device = next(net.parameters()).device
    net.eval()
    with torch.no_grad():
        inputs = torch.as_tensor(encoded, device=device)
        cache = getattr(net, '_checkpoint_inference_state', None)
        if cache is None:
            prediction = net(inputs)
        else:
            # Use the actual fitted frozen values, avoiding device-dependent
            # Transformer recomputation while keeping the full network editable.
            weights = cache['marker_weights'].to(device)
            alpha = cache['alpha'].to(device)
            coefficient = cache['base_coefficient'].to(device)
            base, deep = inputs[:, :net.p_base], inputs[:, net.p_base:]
            prediction = (net.lib_y.mean() + base @ coefficient
                          + net.beta * (net._kdeep(deep, net.lib_deep, weights) @ alpha))
        prediction = prediction.cpu().numpy().astype(np.float64)
    prediction = prediction * transform.sd + transform.mean
    if not np.isfinite(prediction).all():
        raise FloatingPointError('Non-finite full-network prediction')
    return prediction


def main():
    parser = argparse.ArgumentParser(description='Predict phenotypes from a saved SVDKL state.')
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--marker-catalog", type=Path)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--device", default="cpu", help="Device for a full .pt checkpoint; compact .npz uses CPU")
    parser.add_argument("--attention-output", type=Path,
                        help="Optional .npz of all layer/head attention maps; requires a full .pt checkpoint")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Output already exists: {args.output}; choose a new --output path")
    if args.attention_output:
        if args.checkpoint.suffix.lower() not in {'.pt', '.pth'}:
            parser.error('--attention-output requires a genuine full .pt checkpoint')
        if args.attention_output.suffix != '.npz':
            parser.error('--attention-output must end in .npz')
        if args.attention_output.exists() or args.attention_output.is_symlink():
            raise FileExistsError(args.attention_output)
    with np.load(args.data, allow_pickle=False) as z:
        raw, marker_ids, accessions = z["dosage"], z["marker_id"], z["accessions"]
    if accessions.ndim != 1 or len(accessions) != len(raw) or len(np.unique(accessions)) != len(accessions):
        raise ValueError("Each genotype row requires one unique accession")
    with threadpool_limits(limits=1):
        if args.checkpoint.suffix.lower() in {'.pt', '.pth'}:
            from .model import load_full_checkpoint
            net, transform, payload, _ = load_full_checkpoint(
                args.checkpoint, device=args.device, marker_catalog=args.marker_catalog)
            prediction = predict_full_values(net, transform, raw, marker_ids)
            if args.attention_output:
                attention = net.transformer.attention_maps(net.tokens).cpu().numpy()
                args.attention_output.parent.mkdir(parents=True, exist_ok=True)
                with args.attention_output.open('xb') as stream:
                    np.savez_compressed(stream, attention=attention,
                                        deep_marker_ids=transform.input_marker_ids[transform.deep_cols],
                                        context=transform.tokens,
                                        axes=np.asarray(['layer', 'head', 'query_marker', 'key_marker']),
                                        model=np.asarray(payload['case']['model']),
                                        trait=np.asarray(payload['case']['trait']),
                                        fold=np.asarray(payload['case']['fold']))
        else:
            prediction = predict_checkpoint(args.checkpoint, raw, marker_ids, args.marker_catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["accession", "prediction"])
        writer.writerows(zip(accessions, prediction.tolist()))
    print(f"Saved {len(prediction)} predictions to {args.output}")


if __name__ == "__main__":
    main()
