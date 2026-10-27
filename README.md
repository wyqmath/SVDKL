# SVDKL for tomato metabolite prediction

SVDKL predicts tomato metabolite levels from structural variant genotypes using deep kernel learning. It combines genetic similarity across deletion variants with a Transformer-weighted kernel of selected variants. Explore predictions for 36 metabolite traits in Colab, or use the Python code locally to train models and predict new samples.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23669233.svg)](https://doi.org/10.5281/zenodo.23669233)

![SVDKL framework](assets/fig1.jpg)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wyqmath/SVDKL/blob/main/notebooks/SVDKL_quickstart.ipynb)

## Repository structure

```text
SVDKL/
├── README.md
├── assets/
│   ├── fig1.jpg
│   └── traits.tsv
├── notebooks/SVDKL_quickstart.ipynb
└── src/
    ├── prepare_data.py
    ├── model.py
    ├── train.py
    ├── predict.py
    └── score.py
```

## Model sizes

Three model sizes are available on [Zenodo](https://zenodo.org/records/23669233), each supporting the same 36 metabolite traits.

| Model | Parameters | Download |
|---|---:|---|
| SVDKL-0.01M | 9,731 | [Weights](https://zenodo.org/api/records/23669233/files/SVDKL-0.01M.zip/content) |
| SVDKL-4.35M | 4,347,395 | [Weights](https://zenodo.org/api/records/23669233/files/SVDKL-4.35M.zip/content) |
| SVDKL-46.46M | 46,462,019 | [Weights](https://zenodo.org/api/records/23669233/files/SVDKL-46.46M.zip/content) |

## Run locally

Use Python 3.12. Prepare the input data, then train a model or run prediction with pretrained weights.

### Install

```bash
git clone https://github.com/wyqmath/SVDKL.git
cd SVDKL
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.3.5 scipy==1.17.1 threadpoolctl==3.6.0
```

Training environment: Python 3.12.3, PyTorch 2.8.0 with CUDA 12.8, NumPy 2.3.5, SciPy 1.17.1, and threadpoolctl 3.6.0. Training used an NVIDIA RTX 5090. For GPU training, use `--index-url https://download.pytorch.org/whl/cu128` for the PyTorch installation. Colab uses its preinstalled PyTorch/CUDA environment.

### Prepare data

Download the two population files linked in [Data](#data) into `data/`, then run:

```bash
python -m src.prepare_data --vcf data/706.sv.vcf.gz --phenotypes data/pheno_exp_meta.txt --output data/dataset_inputs.npz
```

### Predict

Download a model from the table above and extract its weights:

```bash
python -m zipfile -e SVDKL-0.01M.zip .
```

Predict SlFM0030 with SVDKL-0.01M:

```bash
python -m src.predict --checkpoint weights/SVDKL-0.01M/SlFM0030.pt --data data/dataset_inputs.npz --model SVDKL-0.01M --trait SlFM0030 --device cpu --output predictions_SlFM0030.csv
```

Replace the model and trait ID to use another model or metabolite.

### Train

```bash
python -m src.train --model SVDKL-0.01M --trait SlFM0030 --fold 0 --data data/dataset_inputs.npz --device cuda:0 --output results
```

Use `--device cuda:0` for GPU training or `--device cpu` for CPU training. Set `--trait all` to train all 36 traits. `--fold 0` selects the fixed split described in [Data](#data). Training runs for up to 105 epochs; validation selects the epoch count, then a newly initialized model is fitted using the full training partition.

`--save-optimizer` saves the final optimizer state. `--resume` verifies and skips completed traits; it does not resume an interrupted epoch.

### Evaluate

Load or train the 36 trait models for one model size, then run:

```bash
python -m src.score --results weights --data data/dataset_inputs.npz --model SVDKL-0.01M --output metrics_SVDKL-0.01M.json
```

Evaluation reports Pearson correlation, Spearman correlation, and RMSE and MAE normalized by the training standard deviation, averaged across the 36 traits. Use `--model all` to evaluate all three model sizes.

## Data

The example dataset contains 324 tomato accessions and 36 metabolite traits. Metabolite names and trait IDs are shown in the Colab trait selector. The [trait annotations](assets/traits.tsv) list all 36 names, classes and identification levels.

Genotypes and phenotypes are from Zhou et al. (2022). Metabolite annotations are from Zhu et al. (2018). Predictions use the numerical scale of the source phenotype table.

| Data | Original publication | Download |
|---|---|---|
| Population SV genotypes | Zhou et al. (2022), [Nature](https://doi.org/10.1038/s41586-022-04808-9) | [706.sv.vcf.gz](http://solomics.agis.org.cn/tomato/ftp/genotypes/706.sv.vcf.gz) |
| Integrated phenotypes | Zhou et al. (2022), [Nature](https://doi.org/10.1038/s41586-022-04808-9) | [pheno_exp_meta.txt](http://solomics.agis.org.cn/tomato/ftp/expression/pheno_exp_meta.txt) |
| Metabolite annotations | Zhu et al. (2018), [Cell](https://doi.org/10.1016/j.cell.2017.12.019) | [mmc2.xls](https://ars.els-cdn.com/content/image/1-s2.0-S009286741731499X-mmc2.xls) |
| Measurement identities | Zhu et al. (2018), [Cell](https://doi.org/10.1016/j.cell.2017.12.019) | [mmc5.xlsx](https://ars.els-cdn.com/content/image/1-s2.0-S009286741731499X-mmc5.xlsx) |

### Fixed split

The fixed split uses 259 accessions for training and 65 for testing, with seed `20261012`. Within training, 207 accessions are used for fitting and 52 for validation. Validation selects the training duration, after which the model is fitted again using all training accessions.

### Sample and trait selection

The source cohort contains 332 accessions. Preparation retains the 324 accessions with finite measurements for all 36 selected traits. Five accessions lack all 36 measurements (SL126, SL198, SL227, SL275 and SL439); three lack SlFM0030 (SL237, SL246 and SL251). The panel contains four metabolites from each of nine annotated classes, ordered by ascending trait ID. Identification levels follow the Cell annotation workbook.

### Input format

Numerical preparation reads the population VCF and integrated phenotype table. The output is `data/dataset_inputs.npz`. Inputs retain all 106,362 original deletion (DEL) columns in VCF order. Each genotype is encoded as 0, 1 or 2 copies of the alternate deletion allele, with −1 for missing or unsupported calls. Coordinates use SL5.0 and 1-based VCF positions; variant length is `max(REF length − ALT length, 1)`.

The prepared example input contains `dosage`, `accessions`, `traits`, `phenotypes`, `marker_id`, `chrom`, `pos` and `svlen`. For new samples, provide an NPZ containing `dosage`, `marker_id` and `accessions`, and use the reference dataset separately:

```bash
python -m src.predict --checkpoint weights/SVDKL-0.01M/SlFM0030.pt --data query.npz --reference-data data/dataset_inputs.npz --model SVDKL-0.01M --trait SlFM0030 --output query_predictions.csv
```

Query markers are aligned by ID.

## Method

SVDKL combines a relationship kernel across all training-QC deletion variants with a Transformer-weighted kernel built from 500 variants selected within training. Gaussian process regression combines the two kernels to predict metabolite traits. Quality control (QC) retains variable deletions with minor-allele frequency > 0.05 and missingness < 0.20. Imputation, variant selection and phenotype scaling use training data.

The Transformer uses 19-dimensional variant descriptors and a two-layer weight head. Training minimizes the Gaussian process marginal negative log-likelihood.

## Contact

- Mingyu Bian: [bianmingyu@stu.xju.edu.cn](mailto:bianmingyu@stu.xju.edu.cn)
- Yiquan Wang: [ethan@stu.xju.edu.cn](mailto:ethan@stu.xju.edu.cn)
