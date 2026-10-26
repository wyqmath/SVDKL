import numpy as np
import torch
from torch import nn


def qc_rows(dosage):
    """Keep missing rate < 0.20 and MAF > 0.05, with exact integer boundaries."""
    observed = dosage >= 0
    n = observed.sum(0, dtype=np.int64)
    missing = len(dosage) - n
    total = np.where(observed, dosage, 0).sum(0, dtype=np.int64)
    af = total / np.maximum(n, 1) / 2
    keep = np.flatnonzero(
        (n > 0) & (5 * missing < len(dosage))
        & (10 * np.minimum(total, 2 * n - total) > n)
    )
    return keep, af[keep]


def impute(dosage, keep, af):
    x = dosage[:, keep].astype(np.float64)
    missing = x < 0
    x[missing] = np.broadcast_to(2 * af, x.shape)[missing]
    return x


def rank_markers(x, y, marker_ids, k=500):
    """Rank train correlations, with marker ID resolving ties after 12-digit rounding."""
    xc, yc = x - x.mean(0), y - y.mean()
    sx = np.sqrt((xc ** 2).sum(0))
    valid = sx > 1e-12
    r = np.full(x.shape[1], np.nan)
    r[valid] = (xc[:, valid] * yc[:, None]).sum(0) / (
        sx[valid] * np.sqrt((yc ** 2).sum())
    )
    strength = np.round(np.abs(r), 12)
    candidates = np.flatnonzero(np.isfinite(strength))
    order = candidates[np.lexsort((marker_ids[candidates], -strength[candidates]))][:k]
    return order, strength[order]


def variant_features(metadata, cols):
    """Three SV types, length, 13 chromosomes, and chromosome-relative position."""
    chrom = np.asarray(metadata['chrom']).astype(int)
    pos = np.asarray(metadata['pos'], dtype=np.float64)
    # Normalize against the complete original marker catalogue, before QC/selection.
    cmax = {c: max(float(pos[chrom == c].max()), 1.0) for c in np.unique(chrom)}
    cols = np.asarray(cols)
    if cols.ndim != 1 or not np.issubdtype(cols.dtype, np.integer):
        raise ValueError('Marker columns must be a one-dimensional integer array')
    if not len(cols) or len(np.unique(cols)) != len(cols) or cols.min() < 0 or cols.max() >= len(chrom):
        raise ValueError('Invalid or duplicate marker columns')
    ci = chrom[cols]
    if not ((ci >= 0) & (ci < 13)).all():
        raise ValueError('Chromosome indices must lie in 0..12')
    feature = np.zeros((len(cols), 18), dtype=np.float32)
    svtype = np.asarray(metadata.get('svtype', np.full(len(chrom), 'DEL')))[cols]
    for k, value in enumerate(('DEL', 'INS', 'INV')):
        feature[:, k] = svtype == value
    feature[:, 3] = np.log10(np.maximum(np.asarray(metadata['svlen'], dtype=np.float64)[cols], 1)) / 5.0
    feature[np.arange(len(cols)), 4 + ci] = 1.0
    feature[:, 17] = pos[cols] / np.asarray([cmax[c] for c in ci])
    return feature


class TrainOnlyTransform:
    """Fit QC, imputation, scaling, and supervised selection on one fitting partition."""

    def fit(self, raw_train, y_train, metadata):
        raw_train = np.asarray(raw_train)
        y = np.asarray(y_train, dtype=np.float64)
        if raw_train.ndim != 2 or y.shape != (len(raw_train),):
            raise ValueError('Expected samples-by-markers dosage and one label per sample')
        if not np.isin(raw_train, [-1, 0, 1, 2]).all():
            raise ValueError('Raw dosage must use -1 for missing and 0/1/2 for ALT copies')
        if len(y) < 2 or not np.isfinite(y).all() or y.std(ddof=1) <= 0:
            raise ValueError('Fitting labels must be finite with positive sample SD')
        self.input_marker_ids = np.asarray(metadata['marker_id']).astype(str)
        if self.input_marker_ids.shape != (raw_train.shape[1],):
            raise ValueError('Marker metadata must follow the raw dosage column order')
        self.marker_cols, self.af = qc_rows(raw_train)
        x = impute(raw_train, self.marker_cols, self.af)
        mu, sd = x.mean(0), x.std(0)
        self.b_keep = np.flatnonzero(sd > 1e-12)
        self.b_mu, self.b_sd = mu[self.b_keep], sd[self.b_keep]
        marker_ids = self.input_marker_ids[self.marker_cols]
        order, strength = rank_markers(x, y, marker_ids, k=500)
        if len(order) != 500 or len(self.b_keep) == 0:
            raise ValueError('This protocol requires at least 500 train-variable QC markers')
        # The selected set is fed to the encoder in original input-column order.
        sorted_columns = np.argsort(order)
        self.d_pos, self.deep_abs_r = order[sorted_columns], strength[sorted_columns]
        self.deep_cols = self.marker_cols[self.d_pos]
        self.d_mu = x[:, self.d_pos].mean(0)
        deep_sd = x[:, self.d_pos].std(0)
        self.d_sd = np.where(deep_sd < 1e-6, 1.0, deep_sd)
        features = variant_features(metadata, self.deep_cols)
        self.tokens = np.concatenate([features, self.deep_abs_r[:, None].astype(np.float32)], axis=1)
        self.mean, self.sd = float(y.mean()), float(y.std(ddof=1))
        self.y_standardized = ((y - self.mean) / self.sd).astype(np.float32)
        self.train_encoded = self.encode_imputed(x)
        self.p_base = len(self.b_keep)
        return self

    def encode_imputed(self, x):
        base = (x[:, self.b_keep] - self.b_mu) / self.b_sd
        deep = (x[:, self.d_pos] - self.d_mu) / self.d_sd
        return np.concatenate([base, deep], axis=1).astype(np.float32)

    def transform(self, raw):
        raw = np.asarray(raw)
        if raw.ndim != 2 or raw.shape[1] != len(self.input_marker_ids):
            raise ValueError('Query dosage width differs from the fitted marker catalogue')
        return self.encode_imputed(impute(raw, self.marker_cols, self.af))


class SVWeightTransformer(nn.Module):
    def __init__(self, in_dim, d_model, nhead, n_layers, dropout):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        self.head[-1].bias.data.fill_(0.541)

    def forward(self, context):
        hidden = self.encoder(self.embed(context).unsqueeze(0))
        logits = self.head(hidden.squeeze(0)).squeeze(-1)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError('Nonfinite SV logits')
        # Log-domain softplus normalization preserves ratios without underflow.
        logw = torch.where(logits < -35, logits, nn.functional.softplus(logits.clamp_min(-35)).log())
        return torch.softmax(logw, dim=0) * len(logits)

    @torch.no_grad()
    def attention_maps(self, context):
        """Return evaluation attention as [layer, head, query marker, key marker]."""
        was_training = self.training
        self.eval()
        try:
            hidden = self.embed(context).unsqueeze(0)
            maps = []
            for layer in self.encoder.layers:
                query = layer.norm1(hidden) if layer.norm_first else hidden
                _, attention = layer.self_attn(
                    query, query, query, need_weights=True, average_attn_weights=False)
                maps.append(attention.squeeze(0))
                hidden = layer(hidden)
            return torch.stack(maps)
        finally:
            self.train(was_training)


def _chol(matrix):
    try:
        return torch.linalg.cholesky(matrix)
    except torch.linalg.LinAlgError:
        return torch.linalg.cholesky(matrix + 1e-4 * torch.eye(len(matrix), device=matrix.device, dtype=matrix.dtype))


class SVDKL(nn.Module):
    """All-QC linear kernel plus an attention-weighted top-500 kernel and GP noise."""

    def __init__(self, transform, cfg):
        super().__init__()
        self.p_base = int(transform.p_base)
        n_lib, k_deep = len(transform.y_standardized), len(transform.d_pos)
        self.transformer = SVWeightTransformer(19, cfg['width'], cfg['heads'], cfg['depth'], cfg['dropout'])
        # Preserve the original float64 scalar parameters and float32 neural/buffer state.
        self.log_beta = nn.Parameter(torch.tensor(np.log(cfg['beta_init'])))
        self.log_delta = nn.Parameter(torch.tensor(np.log(cfg['delta_init'])))
        self.register_buffer('tokens', torch.as_tensor(transform.tokens, dtype=torch.float32), persistent=False)
        self.register_buffer('lib_base', torch.zeros(n_lib, self.p_base))
        self.register_buffer('K_base_tt', torch.zeros(n_lib, n_lib))
        self.register_buffer('lib_deep', torch.zeros(n_lib, k_deep))
        self.register_buffer('lib_y', torch.zeros(n_lib))
        self.register_buffer('v_base', torch.zeros(self.p_base))
        self.register_buffer('a_fix', torch.zeros(n_lib))
        self.register_buffer('W_fix', torch.zeros(k_deep))
        self.register_buffer('frozen', torch.zeros((), dtype=torch.long))
        self.register_buffer('kb_ready', torch.zeros((), dtype=torch.long))
        self.lib_base.copy_(torch.from_numpy(transform.train_encoded[:, :self.p_base]))
        self.lib_deep.copy_(torch.from_numpy(transform.train_encoded[:, self.p_base:]))
        self.lib_y.copy_(torch.from_numpy(transform.y_standardized))

    def train(self, mode=True):
        # A further fitting step must never reuse a prior fitted inference cache.
        if mode and hasattr(self, '_checkpoint_inference_state'):
            del self._checkpoint_inference_state
        return super().train(mode)

    @property
    def beta(self):
        return torch.exp(self.log_beta)

    @property
    def delta(self):
        return torch.exp(self.log_delta)

    @staticmethod
    def _kdeep(xa, xb, weights):
        # Direct weighted Gram avoids sqrt(0) gradients while retaining the kernel.
        return (xa * (weights / weights.sum()).unsqueeze(0)) @ xb.T

    def nll(self, inputs):
        if len(inputs) != len(self.lib_y):
            raise ValueError('SVDKL uses one full training batch per epoch')
        if not bool(self.kb_ready):
            with torch.no_grad():
                self.K_base_tt.copy_(self.lib_base @ self.lib_base.T / self.p_base)
                self.kb_ready.fill_(1)
        weights = self.transformer(self.tokens)
        deep = inputs[:, self.p_base:]
        covariance = self.K_base_tt + self.beta * self._kdeep(deep, deep, weights)
        covariance = covariance + self.delta * torch.eye(len(inputs), device=covariance.device)
        residual = self.lib_y - self.lib_y.mean()
        chol = _chol(covariance)
        alpha = torch.cholesky_solve(residual.unsqueeze(-1), chol).squeeze(-1)
        return 0.5 * (residual @ alpha) + torch.log(torch.diag(chol)).sum()

    def _alpha(self, weights):
        if not bool(self.kb_ready):
            raise RuntimeError('The training kernel has not been initialized')
        covariance = self.K_base_tt + self.beta * self._kdeep(self.lib_deep, self.lib_deep, weights)
        covariance = covariance + self.delta * torch.eye(len(covariance), device=covariance.device)
        residual = self.lib_y - self.lib_y.mean()
        return torch.cholesky_solve(residual.unsqueeze(-1), _chol(covariance)).squeeze(-1)

    def forward(self, inputs):
        base, deep = inputs[:, :self.p_base], inputs[:, self.p_base:]
        if bool(self.frozen):
            weights, alpha = self.W_fix, self.a_fix
            base_prediction = base @ self.v_base
        else:
            weights = self.transformer(self.tokens)
            alpha = self._alpha(weights)
            base_prediction = (base @ self.lib_base.T / self.p_base) @ alpha
        return self.lib_y.mean() + base_prediction + self.beta * (self._kdeep(deep, self.lib_deep, weights) @ alpha)

    @torch.no_grad()
    def freeze(self):
        self.eval()
        weights = self.transformer(self.tokens)
        alpha = self._alpha(weights)
        self.v_base.copy_(self.lib_base.T @ alpha / self.p_base)
        self.a_fix.copy_(alpha)
        self.W_fix.copy_(weights)
        self.frozen.fill_(1)
        self.lib_base = self.lib_base.new_zeros(0, self.p_base)

    def compact_state(self, transform):
        if not bool(self.frozen):
            raise RuntimeError('Freeze the model before saving its compact predictor')
        def array(tensor):
            return tensor.detach().cpu().numpy().copy()
        return {
            'kind': np.asarray('svdkl'), 'marker_cols': transform.marker_cols,
            'imputation_mean': 2 * transform.af,
            'b_keep': transform.b_keep, 'b_mu': transform.b_mu, 'b_sd': transform.b_sd,
            'd_pos': transform.d_pos, 'd_mu': transform.d_mu, 'd_sd': transform.d_sd,
            'marker_weights': array(self.W_fix), 'alpha': array(self.a_fix),
            'base_coefficient': array(self.v_base), 'lib_deep': array(self.lib_deep),
            'standardized_intercept': np.asarray(float(self.lib_y.mean())),
            'beta': np.asarray(float(self.beta.detach())), 'delta': np.asarray(float(self.delta.detach())),
            'mean': np.asarray(transform.mean), 'sd': np.asarray(transform.sd),
            'input_marker_ids': transform.input_marker_ids,
            'marker_ids': transform.input_marker_ids[transform.marker_cols],
            'deep_marker_ids': transform.input_marker_ids[transform.deep_cols],
            'context': transform.tokens,
        }


FULL_CHECKPOINT_FORMAT = 'svdkl.full_network.v1'
_TRANSFORM_ARRAYS = (
    'marker_cols', 'af', 'b_keep', 'b_mu', 'b_sd', 'd_pos', 'deep_abs_r',
    'deep_cols', 'd_mu', 'd_sd', 'tokens', 'y_standardized', 'train_encoded',
)


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_copy(item) for item in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f'Unsupported full-checkpoint value: {type(value).__name__}')


def full_checkpoint(net, transform, model_config, training_config, case, optimizer=None):
    """Capture the actual fitted network before freeze discards its training buffer."""
    if bool(net.frozen) or not bool(net.kb_ready):
        raise ValueError('Capture a trained, unfrozen network before calling freeze()')
    fitted = {key: torch.from_numpy(getattr(transform, key).copy()) for key in _TRANSFORM_ARRAYS}
    fitted.update(input_marker_ids=transform.input_marker_ids.tolist(),
                  mean=transform.mean, sd=transform.sd, p_base=transform.p_base)
    return {
        'format': FULL_CHECKPOINT_FORMAT, 'stage': 'completed_outer_refit',
        'state_dict': _cpu_copy(net.state_dict()), 'transform': fitted,
        'model_config': _cpu_copy(model_config), 'training_config': _cpu_copy(training_config),
        'case': _cpu_copy(case), 'torch_version': str(torch.__version__),
        'optimizer_saved': optimizer is not None,
        'optimizer_state_dict': _cpu_copy(optimizer.state_dict()) if optimizer is not None else None,
    }


def load_full_checkpoint(path, device='cpu', restore_optimizer=False, marker_catalog=None):
    """Load genuine full parameters safely; returns model, transform, payload, optimizer.

    The model is returned in evaluation mode. For further fitting, call train(),
    use transform.train_encoded, and select a new learning-rate/training budget.
    Optimizer recovery is available only when it was explicitly saved.
    """
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(payload, dict) and payload.get('format') == INFERENCE_CHECKPOINT_FORMAT:
        if restore_optimizer:
            raise ValueError('This inference release excludes optimizer state; use the training archive to resume fitting')
        return _load_inference_checkpoint(payload, path, device, marker_catalog)
    required = {'format', 'stage', 'state_dict', 'transform', 'model_config',
                'training_config', 'case', 'torch_version', 'optimizer_saved', 'optimizer_state_dict'}
    optional = {'inference_state', 'training_record'}
    if (not isinstance(payload, dict) or not required <= set(payload)
            or set(payload) - required - optional):
        raise ValueError('Expected a complete SVDKL full-network checkpoint')
    if 'training_record' in payload and not isinstance(payload['training_record'], dict):
        raise ValueError('Invalid full-checkpoint training record')
    if payload['format'] != FULL_CHECKPOINT_FORMAT or payload['stage'] != 'completed_outer_refit':
        raise ValueError('Unsupported SVDKL full-network checkpoint format or stage')
    saved = payload['transform']
    if set(saved) != set(_TRANSFORM_ARRAYS) | {'input_marker_ids', 'mean', 'sd', 'p_base'}:
        raise ValueError('Full checkpoint has incomplete fitted preprocessing')
    transform = TrainOnlyTransform()
    for key in _TRANSFORM_ARRAYS:
        if not isinstance(saved[key], torch.Tensor):
            raise ValueError(f'Invalid preprocessing tensor: {key}')
        setattr(transform, key, saved[key].numpy().copy())
    transform.input_marker_ids = np.asarray(saved['input_marker_ids'], dtype=str)
    transform.mean, transform.sd, transform.p_base = saved['mean'], saved['sd'], saved['p_base']
    n_train = len(transform.y_standardized)
    if (transform.tokens.shape != (500, 19) or len(transform.d_pos) != 500
            or transform.train_encoded.shape != (n_train, transform.p_base + 500)
            or not np.isfinite(transform.sd) or transform.sd <= 0
            or not np.isfinite(transform.mean)
            or len(np.unique(transform.input_marker_ids)) != len(transform.input_marker_ids)):
        raise ValueError('Invalid full-checkpoint preprocessing dimensions or phenotype scale')
    if any(not np.isfinite(getattr(transform, key)).all() for key in _TRANSFORM_ARRAYS):
        raise ValueError('Non-finite full-checkpoint preprocessing')
    case = payload['case']
    if (len(case['train_ids']) != n_train or len(set(case['train_ids'])) != n_train
            or set(case['train_ids']) & set(case['held_ids'])):
        raise ValueError('Invalid full-checkpoint training/held accession roles')
    cfg = dict(payload['training_config'], width=payload['model_config']['width'],
               depth=payload['model_config']['depth'])
    # Loading must not consume the caller's random initialization stream.
    with torch.random.fork_rng(devices=[]):
        net = SVDKL(transform, cfg)
    if sum(p.numel() for p in net.parameters()) != payload['model_config']['parameters']:
        raise ValueError('Full-checkpoint parameter count does not match its architecture')
    net.load_state_dict(payload['state_dict'], strict=True)
    if bool(net.frozen) or not bool(net.kb_ready):
        raise ValueError('Full checkpoint must retain the unfrozen training network')
    for name, expected in (
            ('lib_base', transform.train_encoded[:, :transform.p_base]),
            ('lib_deep', transform.train_encoded[:, transform.p_base:]),
            ('lib_y', transform.y_standardized)):
        if not torch.equal(getattr(net, name), torch.from_numpy(expected)):
            raise ValueError(f'Full-checkpoint {name} disagrees with fitted preprocessing')
    if any(not bool(torch.isfinite(value).all()) for value in net.state_dict().values()):
        raise ValueError('Non-finite full-network parameter or buffer')
    if payload['optimizer_saved'] != (payload['optimizer_state_dict'] is not None):
        raise ValueError('Inconsistent optimizer-state declaration')
    inference_state = payload.get('inference_state')
    if 'inference_state' in payload:
        expected_shapes = {
            'marker_weights': (len(transform.d_pos),),
            'alpha': (n_train,),
            'base_coefficient': (transform.p_base,),
        }
        if not isinstance(inference_state, dict) or set(inference_state) != set(expected_shapes):
            raise ValueError('Incomplete full-checkpoint inference cache')
        for name, shape in expected_shapes.items():
            value = inference_state[name]
            if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
                    or value.dtype != net.lib_deep.dtype or value.shape != shape
                    or not bool(torch.isfinite(value).all())):
                raise ValueError(f'Invalid full-checkpoint inference cache: {name}')
        weights = inference_state['marker_weights']
        if (bool((weights < 0).any()) or not bool(torch.isfinite(weights.sum()))
                or not bool(weights.sum() > 0)):
            raise ValueError('Invalid full-checkpoint marker weights')
    net.to(device).eval()
    if inference_state is not None:
        # Retain the real device-of-fit cache without replacing full network state.
        net._checkpoint_inference_state = _cpu_copy(inference_state)
    optimizer = None
    if restore_optimizer:
        if not payload['optimizer_saved']:
            raise ValueError('Optimizer state was not saved; create a new optimizer for fine-tuning')
        optimizer = torch.optim.AdamW(net.parameters(), lr=cfg['peak_lr'], weight_decay=cfg['weight_decay'])
        optimizer.load_state_dict(payload['optimizer_state_dict'])
    return net, transform, payload, optimizer


def load_parameter_checkpoint(path, reference_data, model_id, trait, device='cpu'):
    from .train import get_benchmark, validate_plan

    plan = get_benchmark()
    model_config, training_config = validate_plan(plan, model_id)
    if trait not in plan['traits']:
        raise ValueError('Choose a trait from the fixed 36-trait panel')
    if isinstance(reference_data, dict):
        arrays = reference_data
    else:
        with np.load(reference_data, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
    required = {'dosage', 'accessions', 'traits', 'phenotypes', 'marker_id', 'chrom', 'pos', 'svlen'}
    if not required <= set(arrays):
        raise ValueError('Parameter-only weights require the complete reference input; use --reference-data for custom queries')
    if (arrays['accessions'].tolist() != plan['accessions']
            or arrays['traits'].tolist() != plan['traits']
            or arrays['dosage'].shape != (324, 106362)
            or arrays['phenotypes'].shape != (324, 36)):
        raise ValueError('Reference data must match the fixed 324-accession, 36-trait benchmark')
    parameters = torch.load(path, map_location='cpu', weights_only=True)
    _check_inference_parameters(parameters, model_config, training_config)
    metadata = {key: arrays[key] for key in ('marker_id', 'chrom', 'pos', 'svlen')}
    if 'svtype' in arrays:
        metadata['svtype'] = arrays['svtype']
    fold = plan['fold_partitions'][0]
    train = fold['train']
    column = plan['traits'].index(trait)
    transform = TrainOnlyTransform().fit(arrays['dosage'][train], arrays['phenotypes'][train, column], metadata)
    configuration = dict(training_config, width=model_config['width'], depth=model_config['depth'])
    with torch.random.fork_rng(devices=[]):
        net = SVDKL(transform, configuration)
    with torch.no_grad():
        for name, value in net.named_parameters():
            value.copy_(parameters[name])
        net.to(device).eval()
        net.K_base_tt.copy_(net.lib_base @ net.lib_base.T / net.p_base)
        net.kb_ready.fill_(1)
        net.freeze()
    payload = {'format': 'svdkl.parameters', 'model_config': model_config,
               'case': {'model': model_id, 'trait': trait, 'fold': fold['fold'],
                        'training_seed': fold['training_seed'],
                        'train_ids': [plan['accessions'][i] for i in train],
                        'held_ids': [plan['accessions'][i] for i in fold['held']]}}
    return net, transform, payload, None

# Public inference files retain every learned parameter and the fitted GP predictor.
INFERENCE_CHECKPOINT_FORMAT = 'svdkl.inference_network.v2'
_INFERENCE_PREPROCESS = ('marker_cols', 'imputation_mean', 'b_keep', 'b_mu', 'b_sd',
                        'd_pos', 'd_mu', 'd_sd', 'context', 'mean', 'sd')
_INFERENCE_GP = ('marker_weights', 'alpha', 'base_coefficient', 'lib_deep',
                 'standardized_intercept', 'beta', 'delta')


def _parameter_spec(model_config, network_config):
    with torch.random.fork_rng(devices=[]), torch.device('meta'):
        template = SVWeightTransformer(19, model_config['width'], network_config['heads'],
                                       model_config['depth'], network_config['dropout'])
    spec = {'transformer.' + name: (tuple(value.shape), value.dtype)
            for name, value in template.named_parameters()}
    spec.update(log_beta=((), torch.float64), log_delta=((), torch.float64))
    return spec


def _check_inference_parameters(state, model_config, network_config):
    spec = _parameter_spec(model_config, network_config)
    if not isinstance(state, dict) or set(state) != set(spec):
        raise ValueError('Inference checkpoint must contain every original named parameter')
    for name, (shape, dtype) in spec.items():
        value = state[name]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.dtype != dtype or not bool(torch.isfinite(value).all())):
            raise ValueError(f'Invalid inference parameter: {name}')
    count = sum(value.numel() for value in state.values())
    if count != model_config['parameters']:
        raise ValueError('Inference parameter count differs from the architecture')
    return count


def _catalog(path, marker_catalog=None):
    from pathlib import Path
    if marker_catalog is None:
        if not isinstance(path, (str, Path)):
            raise ValueError('Supply marker_catalog for an in-memory inference checkpoint')
        marker_catalog = next((parent / 'marker_ids.npy' for parent in Path(path).resolve().parents
                               if (parent / 'marker_ids.npy').is_file()), None)
    if marker_catalog is None:
        raise FileNotFoundError('Shared marker_ids.npy is missing; supply --marker-catalog')
    values = np.asarray(np.load(marker_catalog, allow_pickle=False))
    if values.ndim != 1 or values.dtype.kind not in 'US' or len(np.unique(values)) != len(values):
        raise ValueError('Shared marker catalogue must contain unique string IDs')
    return values.astype(str)


def _inference_transform(payload, catalog):
    required = {'format', 'state_dict', 'model_config', 'network_config', 'case',
                'marker_catalog_file', 'marker_count', 'preprocessing', 'gp_state'}
    if set(payload) != required or payload['format'] != INFERENCE_CHECKPOINT_FORMAT:
        raise ValueError('Incomplete inference checkpoint')
    if payload['marker_catalog_file'] != 'marker_ids.npy' or len(catalog) != payload['marker_count']:
        raise ValueError('Shared marker catalogue differs from the inference reference')
    case = payload['case']
    if (set(case) != {'model', 'trait', 'fold', 'training_seed'}
            or case['model'] != payload['model_config']['id']
            or not isinstance(case['trait'], str) or not case['trait']
            or type(case['fold']) is not int or case['fold'] not in range(5)
            or type(case['training_seed']) is not int):
        raise ValueError('Invalid inference case identity')
    pre, gp = payload['preprocessing'], payload['gp_state']
    if set(pre) != set(_INFERENCE_PREPROCESS) or set(gp) != set(_INFERENCE_GP):
        raise ValueError('Incomplete inference preprocessing or GP state')
    for group in (pre, gp):
        if any(not isinstance(v, torch.Tensor) or not bool(torch.isfinite(v).all())
               for v in group.values()):
            raise ValueError('Nonfinite or invalid inference tensor')
    for name in ('marker_cols', 'b_keep', 'd_pos'):
        v = pre[name]
        if v.dtype != torch.int64 or v.ndim != 1 or not len(v) or len(torch.unique(v)) != len(v):
            raise ValueError(f'Invalid inference index: {name}')
    n_qc, p_base = len(pre['marker_cols']), len(pre['b_keep'])
    if (int(pre['marker_cols'].min()) < 0 or int(pre['marker_cols'].max()) >= len(catalog)
            or int(pre['b_keep'].min()) < 0 or int(pre['b_keep'].max()) >= n_qc
            or len(pre['d_pos']) != 500 or int(pre['d_pos'].min()) < 0
            or int(pre['d_pos'].max()) >= n_qc):
        raise ValueError('Inference marker positions are outside the shared catalogue')
    shapes = {'imputation_mean': (n_qc,), 'b_mu': (p_base,), 'b_sd': (p_base,),
              'd_mu': (500,), 'd_sd': (500,), 'context': (500, 19), 'mean': (), 'sd': ()}
    for name, shape in shapes.items():
        dtype = torch.float32 if name == 'context' else torch.float64
        if tuple(pre[name].shape) != shape or pre[name].dtype != dtype:
            raise ValueError(f'Invalid inference preprocessing: {name}')
    if (not bool(pre['sd'] > 0) or not bool((pre['b_sd'] > 0).all())
            or not bool((pre['d_sd'] > 0).all())
            or not bool(((pre['imputation_mean'] >= 0) & (pre['imputation_mean'] <= 2)).all())):
        raise ValueError('Invalid inference scale or imputation value')
    if gp['alpha'].ndim != 1 or not len(gp['alpha']):
        raise ValueError('Invalid GP posterior coefficients')
    gp_shapes = {'marker_weights': (500,), 'alpha': (len(gp['alpha']),),
                 'base_coefficient': (p_base,), 'lib_deep': (len(gp['alpha']), 500),
                 'standardized_intercept': (), 'beta': (), 'delta': ()}
    for name, shape in gp_shapes.items():
        dtype = torch.float64 if name in {'standardized_intercept', 'beta', 'delta'} else torch.float32
        if tuple(gp[name].shape) != shape or gp[name].dtype != dtype:
            raise ValueError(f'Invalid GP inference state: {name}')
    if (bool((gp['marker_weights'] < 0).any()) or not bool(gp['marker_weights'].sum() > 0)
            or not bool(torch.isfinite(gp['marker_weights'].sum()))
            or not bool(gp['beta'] > 0) or not bool(gp['delta'] > 0)):
        raise ValueError('Invalid GP weights or variance parameters')
    _check_inference_parameters(payload['state_dict'], payload['model_config'], payload['network_config'])
    # The fitted scalar is retained exactly; exp on a different device may differ by one ULP.
    for name in ('beta', 'delta'):
        expected = np.exp(float(payload['state_dict']['log_' + name]))
        if not np.isclose(float(gp[name]), expected, rtol=1e-12, atol=0):
            raise ValueError(f'GP {name} differs from the learned log parameter')
    transform = TrainOnlyTransform()
    for name in ('marker_cols', 'b_keep', 'b_mu', 'b_sd', 'd_pos', 'd_mu', 'd_sd'):
        setattr(transform, name, pre[name].numpy().copy())
    transform.af = pre['imputation_mean'].numpy().copy() / 2
    transform.tokens = pre['context'].numpy().copy()
    transform.mean, transform.sd = float(pre['mean']), float(pre['sd'])
    transform.input_marker_ids, transform.p_base = catalog, p_base
    transform.deep_cols = transform.marker_cols[transform.d_pos]
    transform.deep_abs_r = transform.tokens[:, 18].copy()
    return transform


class SVDKLInference(nn.Module):
    """All learned network parameters plus the actual fixed GP predictor, without training state."""
    def __init__(self, transform, model_config, network_config, gp_state):
        super().__init__()
        self.p_base = transform.p_base
        self.transformer = SVWeightTransformer(19, model_config['width'], network_config['heads'],
                                               model_config['depth'], network_config['dropout'])
        self.log_beta = nn.Parameter(torch.zeros((), dtype=torch.float64), requires_grad=False)
        self.log_delta = nn.Parameter(torch.zeros((), dtype=torch.float64), requires_grad=False)
        self.register_buffer('tokens', torch.from_numpy(transform.tokens.copy()), persistent=False)
        for name, value in gp_state.items():
            self.register_buffer('prediction_' + name, value.detach().clone(), persistent=False)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        if mode:
            raise RuntimeError('Inference checkpoints have fixed GP coefficients; load the training archive for fitting')
        return super().train(False)

    @property
    def beta(self):
        return torch.exp(self.log_beta)

    @property
    def delta(self):
        return torch.exp(self.log_delta)

    def forward(self, inputs):
        base, deep = inputs[:, :self.p_base], inputs[:, self.p_base:]
        kernel = SVDKL._kdeep(deep, self.prediction_lib_deep, self.prediction_marker_weights)
        return (self.prediction_standardized_intercept + base @ self.prediction_base_coefficient
                + self.prediction_beta * (kernel @ self.prediction_alpha))


def _load_inference_checkpoint(payload, path, device, marker_catalog):
    transform = _inference_transform(payload, _catalog(path, marker_catalog))
    with torch.random.fork_rng(devices=[]):
        net = SVDKLInference(transform, payload['model_config'], payload['network_config'], payload['gp_state'])
    net.load_state_dict(payload['state_dict'], strict=True)
    net.to(device).eval()
    return net, transform, payload, None


def _build_inference_payload(full_path, compact_path, marker_catalog_path):
    source = torch.load(full_path, map_location='cpu', weights_only=True, mmap=True)
    if source.get('format') != FULL_CHECKPOINT_FORMAT or source.get('stage') != 'completed_outer_refit':
        raise ValueError('Export requires an original complete trained-network checkpoint')
    with np.load(compact_path, allow_pickle=False) as data:
        compact = {key: data[key] for key in data.files}
    catalog = _catalog(full_path, marker_catalog_path)
    np.testing.assert_array_equal(catalog, np.asarray(source['transform']['input_marker_ids'], dtype=str))
    np.testing.assert_array_equal(catalog, compact['input_marker_ids'])
    case = {key: source['case'][key] for key in ('model', 'trait', 'fold', 'training_seed')}
    for key, value in case.items():
        if compact[key].item() != value:
            raise ValueError(f'Original complete and compact checkpoint identities differ: {key}')
    network_config = {key: source['training_config'][key] for key in ('heads', 'dropout')}
    names = _parameter_spec(source['model_config'], network_config)
    state = {key: source['state_dict'][key].detach().clone() for key in names}
    _check_inference_parameters(state, source['model_config'], network_config)
    for key in ('marker_cols', 'b_keep', 'b_mu', 'b_sd', 'd_pos', 'd_mu', 'd_sd'):
        np.testing.assert_array_equal(compact[key], source['transform'][key].numpy())
    np.testing.assert_array_equal(compact['imputation_mean'], 2 * source['transform']['af'].numpy())
    np.testing.assert_array_equal(compact['context'], source['transform']['tokens'].numpy())
    np.testing.assert_array_equal(compact['lib_deep'], source['state_dict']['lib_deep'].numpy())
    np.testing.assert_array_equal(compact['marker_ids'], catalog[compact['marker_cols']])
    np.testing.assert_array_equal(compact['deep_marker_ids'], catalog[compact['marker_cols'][compact['d_pos']]])
    for key in ('mean', 'sd'):
        if float(compact[key]) != source['transform'][key]:
            raise ValueError(f'Original phenotype {key} differs between the saved states')
    if 'inference_state' in source:
        for key in ('marker_weights', 'alpha', 'base_coefficient'):
            np.testing.assert_array_equal(compact[key], source['inference_state'][key].numpy())
    tensor_group = lambda keys: {key: torch.from_numpy(np.asarray(compact[key]).copy()) for key in keys}
    payload = {'format': INFERENCE_CHECKPOINT_FORMAT, 'state_dict': state,
               'model_config': _cpu_copy(source['model_config']), 'network_config': network_config,
               'case': case, 'marker_catalog_file': 'marker_ids.npy', 'marker_count': len(catalog),
               'preprocessing': tensor_group(_INFERENCE_PREPROCESS), 'gp_state': tensor_group(_INFERENCE_GP)}
    _inference_transform(payload, catalog)
    return payload


def _inference_component_bytes(payload):
    size = lambda values: sum(v.numel() * v.element_size() for v in values)
    state = payload['state_dict']
    return {'nn_parameters': size(v for k, v in state.items() if k.startswith('transformer.')),
            'gp_parameters': size(v for k, v in state.items() if k in {'log_beta', 'log_delta'}),
            'gp_inference_state': size(payload['gp_state'].values()),
            'preprocessing': size(payload['preprocessing'].values())}


def export_inference_checkpoint(full_path, compact_path, output_path, marker_catalog_path):
    """Write a lossless inference release; keep the original training archive unchanged."""
    import os
    import tempfile
    from pathlib import Path
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    payload = _build_inference_payload(full_path, compact_path, marker_catalog_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.pending', delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        if temporary is not None and temporary.is_file():
            temporary.unlink()
    return {'status': 'PASS', 'source_bytes': Path(full_path).stat().st_size,
            'exported_bytes': output.stat().st_size,
            'parameter_count': sum(v.numel() for v in payload['state_dict'].values()),
            'bytes_by_component': _inference_component_bytes(payload)}


def verify_inference_checkpoint(full_path, compact_path, exported_path, marker_catalog_path):
    """Compare all released parameters and GP/preprocessing values to their actual source states."""
    expected = _build_inference_payload(full_path, compact_path, marker_catalog_path)
    actual = torch.load(exported_path, map_location='cpu', weights_only=True, mmap=True)
    _inference_transform(actual, _catalog(exported_path, marker_catalog_path))
    if set(actual) != set(expected):
        raise ValueError('Exported inference fields differ')
    for group in ('state_dict', 'preprocessing', 'gp_state'):
        if set(actual[group]) != set(expected[group]):
            raise ValueError(f'Exported {group} keys differ')
        for key, value in expected[group].items():
            other = actual[group][key]
            if other.dtype != value.dtype or other.shape != value.shape or not torch.equal(other, value):
                raise ValueError(f'Exported {group}/{key} differs from the original values')
    for key in set(expected) - {'state_dict', 'preprocessing', 'gp_state'}:
        if actual[key] != expected[key]:
            raise ValueError(f'Exported metadata differs: {key}')
    return {'status': 'PASS', 'parameter_count': sum(v.numel() for v in actual['state_dict'].values()),
            'parameter_tensors': len(actual['state_dict']), 'parameters_exact': True,
            'gp_state_exact': True, 'preprocessing_exact': True, 'catalog_exact': True}
