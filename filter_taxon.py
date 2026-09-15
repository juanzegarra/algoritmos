from Bio import SeqIO
import glob


def filter_taxon(fasta_file):

    seen = set()

    processed_file = fasta_file.replace('.fasta', '_processed.fasta')

    with open(processed_file, 'w') as out:

        for record in SeqIO.parse(fasta_file, 'fasta'):

            sequence = str(record.seq)

            if sequence in seen:
                continue

            seen.add(sequence)

            out.write(
                f'>{record.description}\n'
                f'{sequence}\n'
            )

    print(f'{fasta_file}: {len(seen)} unique sequences')


for file in glob.glob('*.fasta'):
    print(file)
    filter_taxon(file)
