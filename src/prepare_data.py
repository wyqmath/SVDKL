import argparse
import csv
import gzip
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np


EXCLUDED_ACCESSIONS = ("SL126", "SL198", "SL227", "SL237", "SL246", "SL251", "SL275", "SL439")
GT_DOSAGE = {"0/0": 0, "0|0": 0, "0/1": 1, "1/0": 1, "0|1": 1, "1|0": 1, "1/1": 2, "1|1": 2}


@contextmanager
def open_text(path):
    if path.suffix == ".gz":
        with path.open("rb", buffering=4 * 1024 * 1024) as raw:
            with gzip.open(raw, "rt", encoding="utf-8-sig", newline="") as stream:
                yield stream
    else:
        with path.open(encoding="utf-8-sig", newline="", buffering=4 * 1024 * 1024) as stream:
            yield stream


def read_phenotypes(path, accessions, traits):
    selected = {}
    wanted = set(accessions)
    with open_text(path) as stream:
        reader = csv.reader(stream, delimiter="\t")
        header = next(reader)
        required = ["IID", *traits]
        if any(header.count(name) != 1 for name in required):
            raise ValueError("Phenotype input must have one IID column and one column per fixed trait.")
        id_column = header.index("IID")
        columns = [header.index(trait) for trait in traits]
        for row_number, row in enumerate(reader, 2):
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f"Malformed phenotype row {row_number}.")
            accession = row[id_column]
            if accession not in wanted:
                continue
            if accession in selected:
                raise ValueError(f"Duplicate accession at phenotype row {row_number}.")
            values = []
            for column in columns:
                try:
                    values.append(float(row[column]))
                except ValueError:
                    if row[column].strip().lower() not in {"", "na", "nan", "n/a", "."}:
                        raise ValueError(f"Unknown phenotype token at row {row_number}, column {column + 1}.")
                    values.append(float("nan"))
            selected[accession] = values
    if set(selected) != wanted:
        raise ValueError(f"Missing source accessions: {sorted(wanted - set(selected))}")
    return np.asarray([selected[accession] for accession in accessions], dtype=np.float64)


def read_genotypes(path, accessions):
    dosage = np.empty((len(accessions), 106362), dtype=np.int8)
    marker_ids, chromosomes, positions, lengths = [], [], [], []
    source_columns = None
    with open_text(path) as stream:
        for row_number, line in enumerate(stream, 1):
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM\t"):
                samples = line.rstrip("\r\n").split("\t")[9:]
                if len(samples) != len(set(samples)) or not set(accessions) <= set(samples):
                    raise ValueError("VCF sample columns do not contain the fixed source cohort uniquely.")
                source_columns = [9 + samples.index(accession) for accession in accessions]
                source_width = len(samples) + 9
                continue
            if line.startswith("#") or not line.strip():
                continue
            if source_columns is None:
                raise ValueError("VCF has no #CHROM header.")
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != source_width:
                raise ValueError(f"Incorrect VCF field count on row {row_number}.")
            if "SVTYPE=DEL" not in fields[7].split(";"):
                continue
            marker = len(marker_ids)
            if marker == dosage.shape[1]:
                raise ValueError("VCF contains more than the expected 106,362 DEL records.")
            formats = fields[8].split(":")
            if formats.count("GT") != 1:
                raise ValueError(f"Missing or duplicate GT in VCF row {row_number}.")
            gt_column = formats.index("GT")
            calls = []
            for column in source_columns:
                parts = fields[column].split(":")
                calls.append(GT_DOSAGE.get(parts[gt_column], -1) if len(parts) > gt_column else -1)
            dosage[:, marker] = calls
            marker_ids.append(f"{fields[0]}:{fields[1]}:DEL")
            chromosomes.append(fields[0])
            positions.append(int(fields[1]))
            lengths.append(max(len(fields[3]) - len(fields[4]), 1))
    if len(marker_ids) != 106362 or len(set(marker_ids)) != 106362:
        raise ValueError("Expected exactly 106,362 unique ordered DEL markers.")
    return {
        "dosage": dosage,
        "marker_id": np.asarray(marker_ids, dtype="U"),
        "chrom": np.asarray(chromosomes, dtype="U"),
        "pos": np.asarray(positions, dtype=np.int64),
        "svlen": np.asarray(lengths, dtype=np.int64),
    }


def assemble(vcf_path, phenotypes_path):
    from .train import get_benchmark

    plan = get_benchmark()
    accessions = sorted([*plan["accessions"], *EXCLUDED_ACCESSIONS])
    traits = sorted(plan["traits"])
    if len(accessions) != len(set(accessions)) or len(accessions) != 332 or len(set(traits)) != 36:
        raise ValueError("The fixed source cohort or trait panel is inconsistent.")
    phenotype = read_phenotypes(phenotypes_path, accessions, traits)
    complete = np.isfinite(phenotype).all(axis=1)
    kept = np.asarray(accessions, dtype="U")[complete]
    if kept.tolist() != plan["accessions"]:
        raise ValueError("Complete-case accession identities differ from the fixed 324-accession benchmark.")
    arrays = read_genotypes(vcf_path, accessions)
    arrays["dosage"] = arrays["dosage"][complete]
    arrays.update(accessions=kept, traits=np.asarray(traits, dtype="U"), phenotypes=phenotype[complete])
    excluded = {
        accession: [trait for trait, value in zip(traits, values) if not np.isfinite(value)]
        for accession, values, keep in zip(accessions, phenotype, complete) if not keep
    }
    return arrays, excluded


def main():
    data = Path(__file__).resolve().parent.parent / "data"
    parser = argparse.ArgumentParser(description="Prepare fixed SVDKL inputs from the original population VCF and phenotype table.")
    parser.add_argument("--vcf", type=Path, default=data / "706.sv.vcf.gz")
    parser.add_argument("--phenotypes", type=Path, default=data / "pheno_exp_meta.txt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-against", type=Path)
    args = parser.parse_args()
    if args.output is None and args.check_against is None:
        parser.error("Supply --output or --check-against.")
    if args.output and args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    arrays, excluded = assemble(args.vcf, args.phenotypes)
    if args.check_against:
        with np.load(args.check_against, allow_pickle=False) as expected:
            if set(arrays) != set(expected.files):
                raise ValueError("Input snapshot has different array names.")
            for key, value in arrays.items():
                np.testing.assert_array_equal(value, expected[key], err_msg=key)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **arrays)
    print(json.dumps({"status": "PASS", "shape": list(arrays["dosage"].shape),
                      "traits": len(arrays["traits"]), "excluded": excluded,
                      "checked_against_archived_arrays": args.check_against is not None}))


if __name__ == "__main__":
    main()
