from Bio import SeqIO
import sys

from Bio.Seq import Seq

label0 = sys.argv[1]
label1 = sys.argv[2]
output = sys.argv[3]

zero_labels = list(SeqIO.parse(label0, "fasta"))
one_labels = list(SeqIO.parse(label1, "fasta"))

with open(f"{output}_annotation.txt", "w") as fout:
    fout.write(f"ID\t{output}")
    for seq in zero_labels:
        fout.write(f"{seq.id}\t0")
    for seq in one_labels:
        fout.write(f"{seq.id}\t1")
