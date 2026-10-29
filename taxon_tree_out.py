# Rodar BLAST
# Rodar alinhamento múltiplo
# Fazer arvore fasttree

import os
import sys
import subprocess
import re
import time
from Bio import SeqIO

fasta_file = sys.argv[1]

def run_blast(fasta_file):
    blast_output = "blast_output.txt"
    try:
        subprocess.check_output(['blastp', '-query', fasta_file, '-db', 'all_uniprot_conc', "-evalue", "1e-10", "-outfmt", "6", "-out", blast_output])
        return blast_output
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}, blastp failed")
        return None

def fetch_sequences(blast_output):
    sequence_cap = 500
    if not blast_output:
        return
    with open(blast_output, "r") as fin, open("sequence_list.fasta", "w") as fout:
        list_of_ids = set()
        for line in fin:
            fields = line.strip().split("\t")
            if len(fields) < 12:
                continue
            query_id = fields[0]
            subject_id = fields[1]
            if subject_id not in list_of_ids:
                list_of_ids.add(subject_id)
                fout.write(f"{subject_id}\n")
            sequence_cap -= 1
            if sequence_cap == 0:
                break

    subprocess.run(["blastdbcmd", "-db", "all_uniprot_conc", "-entry_batch", "sequence_list.fasta", "-out", "sequence_list.fasta"], shell=True)
    return "sequence_list.fasta"

def run_multiple_alignment(sequence_list):
    famsa_output = "famsa_output.fasta"
    subprocess.run(["famsa", sequence_list, famsa_output], shell=True)
    return famsa_output
