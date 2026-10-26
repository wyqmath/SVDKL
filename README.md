# Deep kernel learning predicts complex traits and prioritizes candidate loci from pangenome structural variation in tomato

SVDKL combines a genomic background kernel with a Transformer-weighted kernel of 500 training-selected variants for Gaussian process prediction. This release contains one fixed-partition experiment for 36 tomato metabolite traits at three model capacities.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wyqmath/SVDKL/blob/main/notebooks/SVDKL_quickstart.ipynb)

![SVDKL framework](assets/fig1.jpg)

## Install

Run commands from the extracted `SVDKL` directory with Python 3.12:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==2.3.5 scipy==1.17.1 threadpoolctl==3.6.0
```

The models were trained with Python 3.12.3, PyTorch 2.8.0/CUDA 12.8, NumPy 2.3.5, SciPy 1.17.1 and threadpoolctl 3.6.0 on an RTX 5090. For CPU prediction, use `https://download.pytorch.org/whl/cpu` in the first command. Colab retains its installed PyTorch/CUDA build and prints the version.

## Release files

`SVDKL_GitHub.zip` contains the source, notebook and framework figure. Choose one capacity ZIP:

| Model ID | Paper label | Width / layers | Parameters | Archive |
|---|---|---|---:|---|
| SVDKL-0.01M | SVDKL-0.01M | 32 / 1 | 9,731 | `SVDKL-0.01M_weights.zip` |
| SVDKL-4.35M | SVDKL-4M | 512 / 2 | 4,347,395 | `SVDKL-4.35M_weights.zip` |
| SVDKL-46.46M | SVDKL-46M | 1,376 / 3 | 46,462,019 | `SVDKL-46.46M_weights.zip` |

Each capacity ZIP contains 75 files: 36 complete inference networks (`.full.pt`) and 36 training records (`.json`) under `weights/rebuilt_fold0/<model>/fold_0/`, one model index, and shared `weights/_input_snapshot.npz` and `weights/marker_ids.npy` files. The directory name `fold_0` identifies the fixed partition used in this experiment.

The inference checkpoints retain all trained neural-network parameters, fitted preprocessing and GP state. They omit AdamW state and duplicate training arrays; complete training checkpoints remain separately archived. Training records describe the original fits and their complete training archives. Keep the extracted `weights` layout so the loader can locate the shared marker catalogue.

The embedded snapshot supplies the data for prediction, training and evaluation. All three ZIPs share the same snapshot and catalogue and can be extracted together. In Colab, use **Existing ZIP path** to open an archive from the runtime or mounted Drive; **Upload ZIP** is also available. Training and scoring are disabled by default.

## Predict

```bash
python -m zipfile -e SVDKL-0.01M_weights.zip .
python -m src.predict --checkpoint weights/rebuilt_fold0/SVDKL-0.01M/fold_0/SVDKL-0.01M_f0_SlFM0030.full.pt --data weights/_input_snapshot.npz --device cpu --output predictions.csv
```

Replace the model and trait IDs to select another network. Prediction aligns markers by ID and exports every supplied accession. The training record identifies the held-out accessions. Custom input NPZ files require `dosage`, `marker_id` and `accessions`; dosage is alternate deletion-allele count 0/1/2, with −1 for missing calls.

## Evaluate

```bash
python -m src.score --results weights/rebuilt_fold0 --data weights/_input_snapshot.npz --model SVDKL-0.01M --output metrics_SVDKL-0.01M.json
```

Evaluation requires the 36 complete networks and their training records for the selected capacity. It uses each trait's held-out accessions, reports per-trait Pearson, Spearman, RMSE/training SD and MAE/training SD, and averages each metric equally across the 36 traits. Training SD uses `ddof=1`; constant predictions receive correlation zero. Use `--model all` after extracting all three capacity ZIPs.

## Train

Fit the 36 traits for one capacity:

```bash
python -m src.train --model SVDKL-0.01M --trait all --fold 0 --data weights/_input_snapshot.npz --device cuda:0 --output results --save-optimizer --resume
```

Use `--trait SlFM0030` for one trait. New training outputs are written to `results/<model>/fold_0/`; `--save-optimizer` includes AdamW state in their complete checkpoints. `--resume` verifies and skips completed cases, rather than continuing an interrupted epoch. Keep these outputs separate from the distributed inference exports. Use their `.full.pt` paths for prediction, or `--results results` for evaluation once all 36 traits are complete.

`src/train.py` fixes the 324 accession IDs, 36 traits, partition and training settings. The experiment uses 259 outer-training and 65 held-out accessions; 207/52 inner training/validation accessions select the refit duration. The seed is `20261012`. A fresh model is then fitted to the outer-training partition. Held-out labels are excluded from fitting and model selection.

Training-only QC retains variable DELs with MAF > 0.05 and missingness < 0.20. All passing DELs form the background kernel; 500 training-selected DELs form the weighted kernel with 19-dimensional descriptors. The Transformer and two-layer weight head optimize GP marginal negative log-likelihood under the fixed 105-epoch schedule. QC, imputation, marker selection and phenotype scaling use the fitting partition at each stage.

## Rebuild inputs from original data

| Source | Original files |
|---|---|
| Zhou et al. (2022), [Nature](https://doi.org/10.1038/s41586-022-04808-9) | [706.sv.vcf.gz](http://solomics.agis.org.cn/tomato/ftp/genotypes/706.sv.vcf.gz), [pheno_exp_meta.txt](http://solomics.agis.org.cn/tomato/ftp/expression/pheno_exp_meta.txt) |
| Zhu et al. (2018), [Cell](https://doi.org/10.1016/j.cell.2017.12.019) | [Annotations: mmc2.xls](https://ars.els-cdn.com/content/image/1-s2.0-S009286741731499X-mmc2.xls), [Measurements: mmc5.xlsx](https://ars.els-cdn.com/content/image/1-s2.0-S009286741731499X-mmc5.xlsx) |

Place the VCF and phenotype table in `data/`. The Cell workbooks document trait identities; preparation reads the two population files:

```bash
python -m src.prepare_data --vcf data/706.sv.vcf.gz --phenotypes data/pheno_exp_meta.txt --output data/dataset_inputs.npz
```

Use the reconstructed file as `--data` in the commands above. Add `--check-against weights/_input_snapshot.npz` to the preparation command to compare all eight arrays with the supplied snapshot.

The fixed 332 paired accessions yield 324 complete cases across the 36 traits. Exclusions are SL126, SL198, SL227, SL237, SL246, SL251, SL275 and SL439. All 106,362 DELs retain source order. Coordinates use SL5.0, 1-based VCF positions; length is `max(REF length − ALT length, 1)`. Phenotypes retain the integrated table's processed units.

<details>
<summary>Trait selection</summary>

The fixed panel has four traits from each of nine annotated classes. Eligibility required ≥150 training, ≥40 validation and ≥20 test observations in each of 30 source partitions (20 random, 10 cross-validation). The identification token `putative` was excluded; `putatively annotated` was retained. Within each class, eligible IDs were ordered by the archived SHA256 of `tomato-benchmark-v1|<trait_id>` (ID breaks ties); the first four were selected. Preparation reuses these 36 IDs in lexicographic order.

</details>
