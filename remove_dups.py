# remove_dups.py
import multiprocessing as mp
from collections import defaultdict

FASTA_FILE = "all_uniprot_concat.fasta"

num_threads = 64

# Global references for Copy-on-Write sharing
seq_lines = []
list_of_seqs = []
renaming_dict = {}
global_keys = []

def init_chunks(fastas):
    global seq_lines
    seq_lines = fastas

def init_list_of_seqs(seq_list):
    global list_of_seqs
    list_of_seqs = seq_list

def init_renaming_dict(dictionary, keys):
    global renaming_dict, global_keys
    renaming_dict = dictionary
    global_keys = keys

def process_file_chunk(start, end):
    """Parses FASTA chunk and returns a set of unique sequences."""
    local_set = set()
    id = None
    annotation = ""
    sequence = ""

    for line in seq_lines[start:end]:
        line = line.strip()
        if not line:
            continue

        if line.startswith(">"):
            # Save the previous accumulated sequence before starting a new one
            if id is not None:
                local_set.add((id, annotation, sequence))

            header = line.split(maxsplit=1)
            id = header[0][1:] # Remove the '>' character
            annotation = header[1] if len(header) > 1 else ""
            sequence = ""
        else:
            if id is not None:
                sequence += line

    # Ensure the last sequence in the chunk is saved
    if id is not None:
        local_set.add((id, annotation, sequence))

    return local_set

def process_list_chunk(start, end):
    small_renaming_dict = defaultdict(list)
    for id, annotation, sequence in list_of_seqs[start:end]:
        small_renaming_dict[id].append((annotation, sequence))
    return small_renaming_dict

def process_renaming_dict(start, end):
    """Processes duplicates and returns a flat list of 3-item tuples."""
    small_renamed_list = []

    for id in global_keys[start:end]:
        items = renaming_dict[id]
        if len(items) < 1:
            print(f"Warning: no sequences found for id {id}")
        elif len(items) > 1:
            for i in range(len(items)):
                # Flattening into 3 items: (id, annotation, sequence)
                small_renamed_list.append((f"{id}_{i}", items[i][0], items[i][1]))
        else:
            small_renamed_list.append((id, items[0][0], items[0][1]))

    return small_renamed_list


def main():
    # 1. Read lines and process file chunks
    with open(FASTA_FILE, "r") as f:
        fastas = f.readlines()

    size = len(fastas)
    chunk_size = size // mp.cpu_count()
    chunks = []
    for i in range(mp.cpu_count()):
        if i == mp.cpu_count() - 1:
            chunks.append((i * chunk_size, size)) # Fixed: Removed -1 to include the end
        else:
            chunks.append((i * chunk_size, i * chunk_size + chunk_size))

    with mp.Pool(mp.cpu_count(), initializer=init_chunks, initargs=(fastas,)) as pool:
        # Collect sets from workers
        sets_from_workers = pool.starmap(process_file_chunk, chunks)

    # Merge all local worker sets into one master set
    set_of_seqs = set()
    for s in sets_from_workers:
        set_of_seqs.update(s)

    # 2. Process list of unique sequences
    unique_list_of_seqs = list(set_of_seqs)
    size_list = len(unique_list_of_seqs)
    chunk_size = size_list // mp.cpu_count() if size_list > 0 else 1

    chunks = []
    for i in range(mp.cpu_count()):
        if i == mp.cpu_count() - 1:
            chunks.append((i * chunk_size, size_list))
        else:
            chunks.append((i * chunk_size, i * chunk_size + chunk_size))

    with mp.Pool(mp.cpu_count(), initializer=init_list_of_seqs, initargs=(unique_list_of_seqs,)) as pool:
        # Fixed: Changed map to starmap because chunks contain tuple arguments (start, end)
        small_dicts = pool.starmap(process_list_chunk, chunks)

    master_renaming_dict = defaultdict(list)
    for small_dict in small_dicts:
        for k, v in small_dict.items():
            master_renaming_dict[k].extend(v)

    # 3. Rename duplicated IDs
    keys = list(master_renaming_dict.keys())
    size_keys = len(keys)
    chunk_size = size_keys // mp.cpu_count() if size_keys > 0 else 1

    chunks = []
    for i in range(mp.cpu_count()):
        if i == mp.cpu_count() - 1:
            chunks.append((i * chunk_size, size_keys))
        else:
            chunks.append((i * chunk_size, i * chunk_size + chunk_size))

    with mp.Pool(mp.cpu_count(), initializer=init_renaming_dict, initargs=(master_renaming_dict, keys)) as pool:
        # Fixed: correct function name and use starmap
        small_lists = pool.starmap(process_renaming_dict, chunks)

    renamed_list = []
    for small_list in small_lists:
        renamed_list.extend(small_list)

    # 4. Write to file
    with open(f"deduplicated_renamed_{FASTA_FILE}", "w") as f:
        for id, annotation, sequence in renamed_list:
            f.write(f">{id} {annotation}\n{sequence}\n")

if __name__ == "__main__":
    main()
