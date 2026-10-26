import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from threadpoolctl import threadpool_limits

from .model import load_full_checkpoint, load_parameter_checkpoint
from .predict import predict_full_values
from .train import get_benchmark, input_snapshot_path, validate_plan

HERE = Path(__file__).resolve().parent.parent
METRICS = ("pcc", "scc", "rmse", "mae")


def validate_input_snapshot(root, arrays):
    snapshot = input_snapshot_path(root)
    if not snapshot.is_file():
        raise FileNotFoundError(f"Missing input snapshot: {snapshot}; supply results with their matching input snapshot")
    with np.load(snapshot, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays):
            raise ValueError("Input array names differ from the results snapshot")
        for key, current in arrays.items():
            previous = saved[key]
            same = previous.shape == current.shape and previous.dtype == current.dtype
            if same:
                same = np.array_equal(previous, current, equal_nan=True) if current.dtype.kind in "fc" else np.array_equal(previous, current)
            if not same:
                raise ValueError(f"Input array {key} differs from the results snapshot")


def metrics(y, prediction, train_sd):
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.shape != prediction.shape or y.ndim != 1 or len(y) < 2:
        raise ValueError("Labels and predictions must have matching one-dimensional shapes")
    if not np.isfinite(y).all() or not np.isfinite(prediction).all() or not np.isfinite(train_sd) or train_sd <= 0:
        raise ValueError("Labels, predictions and positive training SD must be finite")
    if np.ptp(y) == 0:
        raise ValueError("Constant held-out labels: correlations are undefined")
    constant = np.ptp(prediction) == 0
    error = prediction - y
    return {
        "pcc": 0.0 if constant else float(pearsonr(y - y[0], prediction - prediction[0]).statistic),
        "scc": 0.0 if constant else float(spearmanr(y, prediction).statistic),
        "rmse": float(np.sqrt(np.mean(error * error)) / train_sd),
        "mae": float(np.mean(np.abs(error)) / train_sd),
        "constant_prediction": bool(constant),
    }


def find_record(root, model_id, fold, trait):
    root = Path(root)
    stem = f'{model_id}_f{fold}_{trait}'
    folders = dict.fromkeys((root / model_id / f'fold_{fold}', root / f'fold_{fold}',
                            root / model_id, root))
    occupied = []
    for folder in folders:
        paths = (folder / (stem + '.json'), folder / (stem + '.full.pt'))
        if any(path.exists() or path.is_symlink() for path in paths):
            occupied.append(paths)
    if len(occupied) != 1 or not all(path.is_file() for path in occupied[0]):
        raise FileNotFoundError(f'Expected one complete network/record pair: {stem}')
    return occupied[0]


def parameter_files(root, model_id, traits):
    root = Path(root)
    folders = dict.fromkeys((root / model_id, root, root / 'weights' / model_id))
    occupied = [folder for folder in folders if any((folder / (trait + '.pt')).exists() for trait in traits)]
    if not occupied:
        return None
    if len(occupied) != 1:
        raise ValueError(f'Multiple parameter directories found for {model_id}')
    files = [occupied[0] / (trait + '.pt') for trait in traits]
    missing = [path.name for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing parameter files for {model_id}: {missing}')
    return files


def score_parameters(arrays, plan, selected_models, checkpoints):
    if (arrays['accessions'].tolist() != plan['accessions']
            or arrays['traits'].tolist() != plan['traits']):
        raise ValueError('Input identities must match the fixed reference dataset')
    fold = plan['fold_partitions'][0]
    train, held = fold['train'], fold['held']
    cases, summary = [], []
    for model, paths in zip(selected_models, checkpoints):
        validate_plan(plan, model['id'])
        trait_cases = []
        for column, (trait, path) in enumerate(zip(plan['traits'], paths)):
            net, transform, _, _ = load_parameter_checkpoint(path, arrays, model['id'], trait)
            prediction = predict_full_values(net, transform, arrays['dosage'][held], arrays['marker_id'])
            sd = transform.sd
            del net, transform
            row = {'model': model['id'], 'trait': trait, 'fold': fold['fold'],
                   'n_train': len(train), 'n_test': len(held), 'train_sd_ddof1': sd,
                   'prediction_source': 'parameters_with_reconstructed_training_context',
                   **metrics(arrays['phenotypes'][held, column], prediction, sd)}
            trait_cases.append(row)
            cases.append(row)
        summary.append({'model': model['id'], 'parameters': model['parameters'],
                        'fold': fold['fold'], 'traits': len(trait_cases),
                        'mean': {key: float(np.mean([row[key] for row in trait_cases])) for key in METRICS}})
        print(f"Scored {model['id']}: {len(trait_cases)} traits", flush=True)
    return {'aggregation': plan['primary_aggregation'], 'uncertainty': plan['uncertainty'],
            'fold': fold['fold'], 'training_seed': fold['training_seed'],
            'prediction_source': 'parameter_checkpoints', 'rows': summary, 'per_trait': cases}


def score_experiment(root, data_path, plan, selected_models):
    with np.load(data_path, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    checkpoints = [parameter_files(root, model['id'], plan['traits']) for model in selected_models]
    if any(paths is not None for paths in checkpoints):
        if any(paths is None for paths in checkpoints):
            raise ValueError('Select one model when mixing parameter-only and complete checkpoint archives')
        return score_parameters(arrays, plan, selected_models, checkpoints)
    validate_input_snapshot(root, arrays)
    raw, labels = arrays['dosage'], arrays['phenotypes']
    ids, traits, marker_ids = arrays['accessions'].tolist(), arrays['traits'].tolist(), arrays['marker_id']
    if ids != plan['accessions'] or traits != plan['traits'] or labels.shape != (len(ids), len(traits)):
        raise ValueError('Data accessions/traits must match the fixed configuration')
    folds = plan['fold_partitions']
    if len(folds) != 1 or folds[0]['fold'] != 0:
        raise ValueError('Evaluation requires the single fixed fold-0 split')
    fold = folds[0]
    tr, held = fold['train'], fold['held']
    if (set(tr) & set(held) or sorted(tr + held) != list(range(len(ids)))
            or len(tr) != 259 or len(held) != 65 or len(traits) != 36):
        raise ValueError('Invalid fixed accession partition or trait set')
    cases, summary = [], []
    for model in selected_models:
        validate_plan(plan, model['id'])
        trait_cases = []
        for j, trait in enumerate(traits):
            record_path, checkpoint_path = find_record(root, model['id'], 0, trait)
            record = json.loads(record_path.read_text(encoding='utf-8'))
            expected = {'model': model['id'], 'trait': trait, 'fold': 0,
                        'n_parameters': model['parameters'], 'training_seed': fold['training_seed'],
                        'inner_train_ids': [ids[i] for i in fold['inner_train']],
                        'inner_val_ids': [ids[i] for i in fold['inner_val']],
                        'train_ids': [ids[i] for i in tr], 'held_ids': [ids[i] for i in held]}
            for key, value in expected.items():
                if record.get(key) != value:
                    raise ValueError(f'{record_path.name}: {key} does not match configuration')
            if record.get('full_checkpoint') != checkpoint_path.name:
                raise ValueError('Training record does not identify the complete network')
            training = record['training']
            if (training.get('fresh_refit') is not True
                    or training.get('outer_test_used_for_selection') is not False
                    or record.get('test_labels_in_fit_payload') is not False):
                raise ValueError('Record does not follow the outer-training-only refit protocol')
            if (not training['curve'] or training['best_epoch'] != training['refit_epochs']
                    or training['best_epoch'] != len(training['refit_curve'])
                    or training['best_epoch'] != min(training['curve'], key=lambda r: r['validation_nmse'])['epoch']):
                raise ValueError('Recorded refit length differs from the selected inner-validation epoch')
            net, transform, payload, _ = load_full_checkpoint(checkpoint_path, device='cpu')
            for key in ('model', 'trait', 'fold', 'training_seed'):
                if payload['case'].get(key) != expected[key]:
                    raise ValueError(f'Complete checkpoint has a different {key}')
            if any(payload['model_config'].get(k) != model[k] for k in ('id', 'width', 'depth', 'parameters')):
                raise ValueError('Complete checkpoint architecture differs from the fixed model')
            sd, mu = float(labels[tr, j].std(ddof=1)), float(labels[tr, j].mean())
            np.testing.assert_allclose([transform.sd, transform.mean,
                                        record['train_sd_ddof1'], record['train_mean']],
                                       [sd, mu, sd, mu], rtol=0, atol=1e-12)
            restored = predict_full_values(net, transform, raw[held], marker_ids)
            del net, transform, payload
            saved = np.asarray(record['predictions'], dtype=float)
            if saved.shape != restored.shape or not np.isfinite(saved).all():
                raise ValueError('Invalid saved held-out predictions')
            difference = float(np.max(np.abs(restored - saved)) / sd)
            if difference >= 2e-4:
                raise ValueError(f'Checkpoint prediction mismatch: {record_path.name}')
            row = {'model': model['id'], 'trait': trait, 'fold': 0,
                   'n_train': len(tr), 'n_test': len(held), 'train_sd_ddof1': sd,
                   'restore_max_error_over_train_sd': difference,
                   **metrics(labels[held, j], saved, sd)}
            trait_cases.append(row)
            cases.append(row)
        summary.append({'model': model['id'], 'parameters': model['parameters'],
                        'fold': 0, 'traits': len(traits),
                        'mean': {key: float(np.mean([row[key] for row in trait_cases])) for key in METRICS}})
        print(f"Scored {model['id']}: {len(traits)} traits on the fixed split", flush=True)
    return {'aggregation': plan['primary_aggregation'], 'uncertainty': plan['uncertainty'],
            'fold': 0, 'training_seed': fold['training_seed'], 'rows': summary, 'per_trait': cases}


def main():
    parser = argparse.ArgumentParser(description='Score 36 SVDKL traits on the single fixed held-out split.')
    parser.add_argument("--results", type=Path, required=True, help="Training output or extracted weights directory")
    parser.add_argument("--data", type=Path, default=HERE / "data" / "dataset_inputs.npz")
    parser.add_argument("--model", default="all")
    parser.add_argument("--output", type=Path, default=Path("metrics.json"))
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Output already exists: {args.output}; choose a new --output path")
    plan = get_benchmark()
    selected = [m for m in plan["models"] if args.model in ("all", m["id"])]
    if not selected:
        parser.error("Unknown model; use all or a fixed SVDKL model ID")
    with threadpool_limits(limits=1):
        result = score_experiment(args.results, args.data, plan, selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with args.output.open("x", encoding="utf-8") as output:
        output.write(serialized)
    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
