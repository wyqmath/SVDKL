# SVDKL: Deep kernel learning for tomato metabolite prediction

SVDKL predicts tomato metabolite levels from genomic structural variants. It combines a relationship kernel across deletion variants with a Transformer-weighted kernel built from 500 variants selected within the training data. Gaussian process regression brings together genome-wide similarity and trait-associated signals. This repository provides data preparation, model training, pretrained prediction and evaluation for 36 metabolite traits. Three model sizes are available, and the Colab notebook guides you through preparing inputs and making predictions.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wyqmath/SVDKL/blob/main/notebooks/SVDKL_quickstart.ipynb)

Choose a model size, load its weight ZIP, and provide the source data or a reference NPZ built with `src.prepare_data`. Run the cells in order to produce a prediction CSV. Training and evaluation have separate controls.

![SVDKL framework](assets/fig1.jpg)

## Repository structure

```text
SVDKL/
├── README.md
├── assets/fig1.jpg
├── notebooks/SVDKL_quickstart.ipynb
└── src/
    ├── prepare_data.py
    ├── model.py
    ├── train.py
    ├── predict.py
    └── score.py
```

## Install

Clone the repository and enter its root directory:

```bash
git clone https://github.com/wyqmath/SVDKL.git
cd SVDKL
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.3.5 scipy==1.17.1 threadpoolctl==3.6.0
```

These commands install CPU PyTorch for prediction. For GPU training, install the CUDA build with `--index-url https://download.pytorch.org/whl/cu128` instead. The training records identify Python 3.12.3, PyTorch 2.8.0/CUDA 12.8, NumPy 2.3.5, SciPy 1.17.1 and threadpoolctl 3.6.0 on an RTX 5090. Colab uses its installed PyTorch/CUDA build and displays the active versions.

## Prepare the data

Download the population VCF and integrated phenotype table from their original publications:

| Data | Original publication | File |
|---|---|---|
| Population SV genotypes | Zhou et al. (2022), [Nature](https://doi.org/10.1038/s41586-022-04808-9) | [706.sv.vcf.gz](http://solomics.agis.org.cn/tomato/ftp/genotypes/706.sv.vcf.gz) |
| Integrated phenotypes | Zhou et al. (2022), [Nature](https://doi.org/10.1038/s41586-022-04808-9) | [pheno_exp_meta.txt](http://solomics.agis.org.cn/tomato/ftp/expression/pheno_exp_meta.txt) |
| Metabolite annotations | Zhu et al. (2018), [Cell](https://doi.org/10.1016/j.cell.2017.12.019) | [mmc2.xls](https://ars.els-cdn.com/content/image/1-s2.0-S009286741731499X-mmc2.xls) |
| Measurement identities | Zhu et al. (2018), [Cell](https://doi.org/10.1016/j.cell.2017.12.019) | [mmc5.xlsx](https://ars.els-cdn.com/content/image/1-s2.0-S009286741731499X-mmc5.xlsx) |

Place `706.sv.vcf.gz` and `pheno_exp_meta.txt` in `data/`, then run:

```bash
python -m src.prepare_data --vcf data/706.sv.vcf.gz --phenotypes data/pheno_exp_meta.txt --output data/dataset_inputs.npz
```

The script selects the fixed 332-accession source cohort and 36 phenotype columns, retaining the 324 accessions with finite measurements for every selected trait. SL126, SL198, SL227, SL275 and SL439 lack all 36 measurements; SL237, SL246 and SL251 lack SlFM0030. The Cell workbooks document the trait identities; numerical preparation reads the two population files.

All 106,362 original DEL columns remain in VCF order. Genotypes are ALT deletion-allele counts 0/1/2, with −1 for missing or unsupported calls. Coordinates use SL5.0 and 1-based VCF positions; variant length is `max(REF length − ALT length, 1)`. Phenotypes retain the processed units of the integrated source table. QC, imputation and scaling are fitted within training partitions.

<details>
<summary>How the 36 traits were selected</summary>

The panel contains four traits from each of nine annotated classes. Eligibility required at least 150 training, 40 validation and 20 test observations in each of 30 historical partitions: 20 random and 10 cross-validation partitions. The identification token `putative` was excluded; `putatively annotated` was retained. Within each class, eligible IDs were ordered by the archived SHA256 ordering of `tomato-benchmark-v1|<trait_id>`, with trait ID breaking ties, and the first four were selected. Preparation reuses the fixed IDs in lexicographic order and performs no new selection or sampling.

</details>

## Pretrained models

Choose the weight ZIP for one model size:

| Model ID | Paper label | Width / layers | Parameters | Weight archive |
|---|---|---|---:|---|
| SVDKL-0.01M | SVDKL-0.01M | 32 / 1 | 9,731 | `SVDKL-0.01M_weights.zip` |
| SVDKL-4.35M | SVDKL-4M | 512 / 2 | 4,347,395 | `SVDKL-4.35M_weights.zip` |
| SVDKL-46.46M | SVDKL-46M | 1,376 / 3 | 46,462,019 | `SVDKL-46.46M_weights.zip` |

Each ZIP stores 36 parameter files, one per trait:

```text
weights/
└── SVDKL-0.01M/
    ├── SlFM0030.pt
    ├── SlFM0047.pt
    └── ...
```

The files contain the trained Transformer and kernel parameters. The predictor uses the reference NPZ to recover the original training preprocessing, selected markers and GP conditioning, then loads the saved parameters. It runs no optimization steps. The weight ZIP contains no input dataset.

## Predict

Extract one weight archive in the repository root and predict a trait:

```bash
python -m zipfile -e SVDKL-0.01M_weights.zip .
python -m src.predict --checkpoint weights/SVDKL-0.01M/SlFM0030.pt --data data/dataset_inputs.npz --model SVDKL-0.01M --trait SlFM0030 --device cpu --output predictions_SlFM0030.csv
```

The CSV contains one prediction per supplied accession. Replace the model and trait IDs to use another saved model. For new accessions, provide an NPZ containing `dosage`, `marker_id` and `accessions`, and retain the fixed reference data separately:

```bash
python -m src.predict --checkpoint weights/SVDKL-0.01M/SlFM0030.pt --data query.npz --reference-data data/dataset_inputs.npz --model SVDKL-0.01M --trait SlFM0030 --output query_predictions.csv
```

Query markers are aligned by ID. Existing output files are preserved; choose another output filename to rerun a command.

## Evaluate

After loading all 36 weight files for a model, evaluate its held-out predictions:

```bash
python -m src.score --results weights --data data/dataset_inputs.npz --model SVDKL-0.01M --output metrics_SVDKL-0.01M.json
```

Evaluation uses the 65 held-out accessions defined in `src/train.py`. It reports Pearson correlation, Spearman correlation, RMSE/training SD and MAE/training SD for each trait, then averages each metric equally across the 36 traits. Training SD uses `ddof=1`; constant predictions receive correlation zero. Use `--model all` after extracting all three capacity ZIPs.

The prediction and evaluation commands also accept the earlier complete `.full.pt` checkpoints with their matching training records and shared inputs. Their saved layout remains supported.

## Train

Train one trait:

```bash
python -m src.train --model SVDKL-0.01M --trait SlFM0030 --fold 0 --data data/dataset_inputs.npz --device cuda:0 --output results --save-optimizer --resume
```

Set `--trait all` to train all 36 traits. In Colab, select a GPU runtime before enabling the training cell. New outputs are written to `results/<model>/fold_0/`. The `--resume` option validates and skips completed cases; it does not resume a partially completed epoch.

The included configuration uses 259 training and 65 held-out accessions. Within the training set, 207/52 inner training/validation accessions select the refit duration, followed by a fresh fit on all 259 training accessions. Roles and seed `20261012` are fixed. This configuration corresponds to the supplied 36-trait parameter files; the paper's five-fold benchmark is a separate experiment.

Training-only QC keeps variable DELs with MAF > 0.05 and missingness < 0.20. All passing DELs form the background kernel; 500 training-selected DELs form the weighted kernel using 19-dimensional descriptors. The Transformer and two-layer weight head optimize GP marginal negative log-likelihood under the fixed 105-epoch schedule. QC, imputation, marker selection and phenotype scaling use the fitting partition at each stage.
