import os
import sys
import time
import subprocess
from pathlib import Path


def run_command(command):
    """Run a command and stop if it fails."""
    print("Running:", " ".join(map(str, command)))

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}:")
        print(" ".join(map(str, command)))
        raise


def run_blast(fasta_file, out_dir, blast_db="/home/juanzegarra71/blastp/dedup_all_uniprot_conc"):
    input_name = Path(fasta_file).stem
    blast_output = out_dir / f"{input_name}_blast_output.txt"

    start = time.perf_counter()

    command = [
        "blastp",
        "-query", str(fasta_file),
        "-db", blast_db,
        "-evalue", "1e-10",
        "-outfmt", "6",
        "-out", str(blast_output),
        "-num_threads", "32",
    ]

    run_command(command)

    elapsed = time.perf_counter() - start

    return blast_output, elapsed


def fetch_sequences(
    blast_output,
    out_dir,
    blast_db="/home/juanzegarra71/blastp/dedup_all_uniprot_conc",
    sequence_cap=500
):
    input_name = Path(blast_output).stem

    sequence_ids_file = out_dir / f"{input_name}_ids.txt"
    sequence_output = out_dir / f"{input_name}_sequences.fasta"

    start = time.perf_counter()

    subject_ids = set()

    with open(blast_output, "r", encoding="utf-8") as fin:
        for line in fin:
            fields = line.rstrip("\n").split("\t")

            if len(fields) < 2:
                continue

            subject_id = fields[1].strip()

            if subject_id:
                subject_ids.add(subject_id)

            if len(subject_ids) >= sequence_cap:
                break

    if not subject_ids:
        raise RuntimeError(
            f"No subject IDs were found in {blast_output}"
        )

    with open(sequence_ids_file, "w", encoding="utf-8") as fout:
        for subject_id in sorted(subject_ids):
            fout.write(f"{subject_id}\n")

    command = [
        "blastdbcmd",
        "-db", blast_db,
        "-entry_batch", str(sequence_ids_file),
        "-out", str(sequence_output),
    ]

    run_command(command)

    elapsed = time.perf_counter() - start

    return sequence_output, elapsed


def run_multiple_alignment(sequence_file, out_dir):
    input_name = Path(sequence_file).stem
    alignment_output = out_dir / f"{input_name}_aln.fasta"

    start = time.perf_counter()

    command = [
        "famsa",
        str(sequence_file),
        str(alignment_output),
        "-t", "16",
        "-refine_mode", "auto",
    ]

    run_command(command)

    elapsed = time.perf_counter() - start

    return alignment_output, elapsed


def trim_alignment(alignment_file, out_dir):
    input_name = Path(alignment_file).stem
    trimmed_output = out_dir / f"{input_name}_trimmed.fasta"

    start = time.perf_counter()

    command = [
        "trimal",
        "-in", str(alignment_file),
        "-out", str(trimmed_output),
        "-strictplus",
    ]

    run_command(command)

    elapsed = time.perf_counter() - start

    return trimmed_output, elapsed


def run_tree_building(trimmed_alignment, out_dir):
    input_name = Path(trimmed_alignment).stem
    tree_output = out_dir / f"{input_name}.nwk"

    start = time.perf_counter()

    command = [
        "fasttree",
        str(trimmed_alignment),
    ]

    with open(tree_output, "w", encoding="utf-8") as fout:
        subprocess.run(
            command,
            stdout=fout,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

    elapsed = time.perf_counter() - start

    return tree_output, elapsed


def main():
    if len(sys.argv) != 3:
        print(
            f"Usage: python {Path(sys.argv[0]).name} "
            "<input_fasta> <output_directory>"
        )
        sys.exit(1)

    fasta_file = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()

    if not fasta_file.exists():
        raise FileNotFoundError(
            f"Input FASTA does not exist: {fasta_file}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()

    blast_output, time_blast = run_blast(
        fasta_file,
        out_dir
    )

    sequence_file, time_fetch = fetch_sequences(
        blast_output,
        out_dir
    )

    alignment_output, time_align = run_multiple_alignment(
        sequence_file,
        out_dir
    )

    trimmed_alignment, time_trim = trim_alignment(
        alignment_output,
        out_dir
    )

    tree_output, time_tree = run_tree_building(
        trimmed_alignment,
        out_dir
    )

    time_total = time.perf_counter() - total_start

    input_name = fasta_file.stem
    time_log = out_dir / f"time_log_{input_name}.txt"

    with open(time_log, "w", encoding="utf-8") as fout:
        fout.write(f"input: {fasta_file}\n")
        fout.write(f"blast_output: {blast_output}\n")
        fout.write(f"sequence_file: {sequence_file}\n")
        fout.write(f"alignment_output: {alignment_output}\n")
        fout.write(f"trimmed_alignment: {trimmed_alignment}\n")
        fout.write(f"tree_output: {tree_output}\n\n")

        fout.write(f"blast: {time_blast:.4f}\n")
        fout.write(f"fetch: {time_fetch:.4f}\n")
        fout.write(f"align: {time_align:.4f}\n")
        fout.write(f"trim: {time_trim:.4f}\n")
        fout.write(f"tree: {time_tree:.4f}\n")
        fout.write(f"total: {time_total:.4f}\n")

    print("\nPipeline completed successfully.")
    print(f"Tree: {tree_output}")
    print(f"Timing log: {time_log}")


if __name__ == "__main__":
    main()
