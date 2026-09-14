import argparse
from operator import methodcaller
from Bio import SeqIO
import glob
from multiprocessing import Pool
from numpy.random.mtrand import wald
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.anno.uniprot_gobases import UniProtGoBase
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import pdist, squareform
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from Bio import Phylo
from Bio.Phylo.Consensus import majority_consensus, strict_consensus
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
from scipy.sparse import eye, bmat
from scipy.sparse.linalg import spsolve
import pandas as pd
import networkx as nx
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='File with fasta sequences')
    parser.add_argument('--help', '-h', type='str', help='print this help message')
    parser.add_argument('--anno', '-a', type='str', help='Annotation file, tab-separated, with first row as header and first column as sequence ID, traits must be labeled as 0s or 1s.')
    parser.add_argument('--phylo', action='store_true', help='Build phylogenetic tree from sequence distances')
    parser.add_argument('--bootstraps', '-b', type=int, default=0,help='Number of bootstraps to perform')
    parser.add_argument('--kmer','-k', type=int, default=3, help='k-mer size for feature table building')
    parser.add_argument('--outdir', type='str', help='Output directory')
    parser.add_argument('--tree-cutoff', type='float', default=0.8, help='Tree cutoff value for defining best neighbor join tree from bootstraps')
    parser.add_argument('--classify_nodes', action='store_true', help='Classify internal nodes using logistic regression')
    parser.add_argument('--processes', '-p', type=int, default=1, help='Number of processes to use')
    parser.add_argument('--assemble_kmers', '-a', action='store_true', type=int, help='Assemble kmers from sequence data')

    return parser.parse_args()

class seqObject:
    def __init__(self, id, seq):
        self.id = id
        self.seq = seq
        self.hash = None #coloca no main para criar como lista se prog for True
        self.kmer_vector = None #colocar no main para criar como lista se prog for True
        self.anno = None


    def kmerize(self, fasta, k):
        kmer_vector = np.zeros(8000)
        for i in range(len(fasta) - k + 1):
            kmer = fasta[i:i+k]
            p = criapos(kmer)
            if p != -1:
                kmer_vector[p - 1] += 1
            else:
                print(f"Warning: invalid character in kmer of {self.id}: {fasta[i:i+k]}")
        self.kmer_vector = kmer_vector

    def hash_seqs(self, k):
        hash_dict = {i: self.seq[i:i+k] for i in range(len(self.seq)-k+1)}
        return hash_dict


    def annotate(self, id, line):
        parts = line.strip().split('\t')
        if line[0] == id:
            self.anno = parts[1:]


def read_annotations(anno_file, sequences):
    annotation_ids = []

    with open(anno_file, 'r') as f:
        header = True
        for line in f:
            parts = line.strip().split('\t')
            if header:
                header = False
                annotation_ids.append(parts[0])
            else:
                for seq in sequences:
                    if seq.id == parts[0]:
                        seq.annotation = parts[1:]
                        break


        return annotation_ids

def create_annotation_vector(sequences):
    create_annotation_vector = np.zeros((len(sequences), len(sequences[0].annotation)))
    for i, seq in enumerate(sequences):
        create_annotation_vector[i] = seq.annotation
    return create_annotation_vector

def read_fasta(fasta_file):
    try:
        sequences = []
        for record in SeqIO.parse(fasta_file, 'fasta'):
            q = seqObject(record.id, record.seq)
            sequences.append(q)
    except Exception as e:
        print("Fasta file error: " + str(e))
        return None
    return sequences


def criapos(janela):
    alfabeto = "ADQIMSYRCGLFTVNEHKPWX"
    pos = {aa: i + 1 for i, aa in enumerate(alfabeto)}

    try:
        i = pos[janela[0]]
        j = pos[janela[1]]
        k = pos[janela[2]]
    except KeyError:
        return -1

    if i <= 20 and j <= 20 and k <= 20:
        return 400 * (i - 1) + 20 * (j - 1) + k

    return -1



def montaA(sequencias):
    A = np.zeros((len(sequencias), 8000))
    for i, seq in enumerate(sequencias):
        A[i] = seq.kmer_vector
    return A

def singular(A, out_dir):
    # 1. Set n_components to an upper bound (e.g., total features or an arbitrary cap)
    max_components = min(A.shape[0], A.shape[1]) - 1
    svd = TruncatedSVD(n_components=max_components, random_state=42)
    svd.fit(A)

    # 2. Calculate cumulative explained variance ratio
    cumulative_variance = np.cumsum(svd.explained_variance_ratio_)

    # 3. Find the minimum k where cumulative variance >= 0.70
    target_variance = 0.70
    optimal_k = np.argmax(cumulative_variance >= target_variance) + 1

    # 4. Refit or slice the transformed data using optimal k
    svd_optimal = TruncatedSVD(n_components=optimal_k, random_state=42)
    A_reduced = svd_optimal.fit_transform(A)

    s = svd.singular_values_
    sum_s2_total = np.sum(s**2)
    sum_s2_top3 = np.sum(s[:3]**2)

    # Calculate exact multiplier for top 3 singular values
    target_variance = 0.70
    alpha = np.sqrt((target_variance * sum_s2_total) / sum_s2_top3)

    # 4. Project original matrix X onto top 3 basis components (V_3)
    # components_ has shape (n_components, n_features)
    V_3 = svd.components_[:3, :]
    A_3D = A @ V_3.T

    # Apply variance scaling factor to sample coordinates
    A_pca_scaled = A_3D * alpha

    # 5. Verify top 3 variance ratio after scaling
    s_scaled = s.copy()
    s_scaled[:3] *= alpha
    top3_ratio = np.sum(s_scaled[:3]**2) / sum_s2_total

    #plot pca

    plt.figure(figsize=(8, 6))
    plt.scatter(A_pca_scaled[:, 0], A_pca_scaled[:, 1], A_pca_scaled[:, 2], c=None, cmap='viridis')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.zlabel('PC3')
    plt.title('PCA Scaled Sample Coordinates')
    plt.colorbar()
    plt.savefig(f"{out_dir}/pca_scaled.png")
    plt.close()


    # Create matrix with reduced features

    A_k = svd.transform(A)[:, :optimal_k]

    with open(f"{out_dir}/svd_log.txt", "w") as f:
        f.write(f"Optimal k: {optimal_k}\n")
        f.write(f"Cumulative variance: {cumulative_variance}\n")
        f.write(f"X_reduced shape: {A_k.shape}\n")
        f.write(f"Top 3 ratio: {top3_ratio}\n")

    return A_reduced, A_k

def kmeans(A_reduced, optimal_k, out_dir):
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(A_reduced)
    centroids = kmeans.cluster_centers_

    plt.scatter(A_reduced[:, 0], A_reduced[:, 1], c=kmeans.labels_)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, c='red')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('KMeans Clustering')
    plt.savefig(f"{out_dir}/kmeans.png")
    plt.close()

def create_sample_index(sequences):
    sample_index = {seq.id : seq for seq in sequences}
    return sample_index

def neighbor_join_reduced(A, sample_labels):
    target_variance = 0.70
    max_components = min(A.shape[0], A.shape[1]) - 1
    svd = TruncatedSVD(n_components=max_components, random_state=42)
    svd.fit(A)

    cumulative_variance = np.cumsum(svd.explained_variance_ratio_)
    optimal_k = np.argmax(cumulative_variance >= target_variance) + 1

    # Refit or slice the transformed data using optimal k
    svd_optimal = TruncatedSVD(n_components=optimal_k, random_state=42)
    A_k = svd_optimal.fit_transform(A)[:, :optimal_k]
    dist_condensed = pdist(A_k, metric='euclidean')
    square_form = squareform(dist_condensed)
    trig_lower = np.tril(square_form, k=-1)
    bio_dm = DistanceMatrix(names=sample_labels, matrix=trig_lower)
    constructor = DistanceTreeConstructor()
    tree = constructor.nj(bio_dm)

    return tree

def bootstrap(A, n_boostrap, sequences, tree_cutoff, out_dir):
    trees = []
    sample_labels = [i.seq for i in sequences]
    for _ in range(n_boostrap):
        A_bootstrap = A[np.random.choice(A.shape[0], A.shape[0], replace=True)]
        sample_labels_bootstrap = sample_labels[np.random.choice(sample_labels.shape[0], sample_labels.shape[0], replace=True)]
        tree = neighbor_join_reduced(A_bootstrap, sample_labels_bootstrap)
        trees.append(tree)

    # Iterate over all bootstrap trees and return the most common node

    tree = majority_consensus(trees, cutoff=tree_cutoff)
    tree.root.confidence = None

    fig, ax = plt.subplots(figsize=(10, 8))


    Phylo.draw(tree, axes=ax, do_show=False)
    plt.savefig("tree_image.png", dpi=300, bbox_inches="tight")
    plt.close()
    Phylo.write(tree, os.path.join(out_dir, "majority_consensus_tree.nwk"), "newick")

    return tree


def logistica(A, indicadores):
    b = np.where(indicadores == 1, 12.0, -12.0)
    return resolve2(A, b)


def resolve2(A, b):
    m, n = A.shape

    In = eye(n, format="csr")
    Im = eye(m, format="csr")

    bm = np.concatenate([
        np.zeros(n),
        b
    ])

    M = bmat([
        [In, A.T],
        [A, -Im]
    ], format="csr")

    x = spsolve(M, bm)

    return x[:n]

def classify_internal_nodes(tree, A):

    taxa = [clade.name for clade in tree.get_terminals()]

    labels = {}

    for node in tree.get_nonterminals():
        descendants = {clade.name for clade in node.get_terminals()}

        vector = np.array([
            1 if taxon in descendants else 0
            for taxon in taxa
        ])

        labels[node] = vector

        weight_dict = {node : weight for node, weight in zip(labels.keys(), logistica(A, labels[node]))}

    return weight_dict, labels

# Get the 10 biggest and 10 lowest weights,
def reduce_weights(A, weight_vector, indicadores):

    lowest_idx = np.argsort(weight_vector)[:10]
    highest_idx = np.argsort(weight_vector)[-10:][::-1]

    selected_idx = np.concatenate([
        lowest_idx,
        highest_idx
    ])

    A_selected = A[:, selected_idx]

    b = np.where(indicadores == 1, 12.0, -12.0)

    weight_vector_selected = logistica(A_selected, b)

    lowest_idx = np.argsort(weight_vector_selected)[:10]
    highest_idx = np.argsort(weight_vector_selected)[-10:][::-1]

    selected_idx = np.concatenate([
        lowest_idx,
        highest_idx
    ])

    return [weight_vector_selected, selected_idx]

def inversa(endereco):
    # Compute the sequence corresponding to the address
    alfabeto = "ADQIMSYRCGLFTVNEHKPWX"

    p = [0, 0, 0]
    k = endereco

    # First position
    p[0] = int(k // 400)

    if (k - p[0] * 400) > 1.00e-10:
        p[0] += 1

    k = k - (p[0] - 1) * 400

    # Second position
    p[1] = int(k // 20)

    if (k - p[1] * 20) > 1.00e-10:
        p[1] += 1

    k = k - (p[1] - 1) * 20

    # Third position
    p[2] = k

    if p[2] == 0:
        p[2] = 1

    # MATLAB uses 1-based indexing, Python uses 0-based indexing
    s = ""

    for i in range(3):
        s += alfabeto[int(p[i]) - 1]

    return s
'''
def assemble_kmers(important_kmers, sequences, k):
    q = nx.DiGraph()
    kmer_counts = defaultdict(int)
    graph_dict = defaultdict(list)

    for seq in sequences:
        for kmer in important_kmers:
            kmer_counts[kmer] += seq.seq.count(kmer)

    for i, kmer in enumerate(important_kmers):
        graph_dict[kmer[:-k]].append((kmer[-k:], i))
'''


def main():
    args = parse_args()
    annotations = args.annotations
    input_sequences = read_fasta(annotations)

    bootstrap_value = int(args.bootstrap)
    out_dir = args.out_dir
    go_annotation = args.go_annotation
    k = int(args.k)
    processes = int(args.processes)
    phylo = args.phylo
    tree_cutoff = float(args.tree_cutoff)

    os.makedirs(out_dir, exist_ok=True)

    #Sanity check reading sequences
    if input_sequences is None:
        print("No sequences found. Exiting.")
        return

    sequences = read_fasta(input_sequences)
    if sequences is None:
        print("Error reading sequences. Exiting...")
        return

    # Sanity check reading annotations
    if annotations is None:
        print("No annotations found. Exiting...")
        return
    else:
        annotation_ids = read_annotations(annotations, sequences)
        annotation_vector = create_annotation_vector(sequences)

    # Create k-mer count vector for each sequence
    if processes > 1 and __name__ == '__main__':
        with Pool(processes=processes) as pool:
            runner = methodcaller('kmerize', k)
            pool.map(runner, sequences)
    elif processes == 1:
        for seq in sequences:
            seq.kmerize(seq, k)


    # Create sample index
    sample_index = create_sample_index(sequences)

    # Create feature matrix
    A = montaA(sequences)

    # Run svd and truncate matrix

    A_reduced, A_3D = singular(A, out_dir)

    # Save reduced matrix
    np.save(f"{out_dir}/A_reduced.npy", A_reduced)
    np.save(f"{out_dir}/A_3D.npy", A_3D)

    if phylo and bootstrap_value == 0:
        sample_labels = [seq.id for seq in sequences]
        tree = neighbor_join_reduced(A, sample_labels)
        string_buffer = io.StringIO()
        Phylo.write(tree, string_buffer, "newick")
        newick_string = string_buffer.getvalue().strip()
        with open(f"{out_dir}/tree.newick", "w") as f:
            f.write(newick_string)
    elif phylo and bootstrap_value > 0:
        tree = bootstrap(A, bootstrap_value, sequences, tree_cutoff, out_dir)

    weight_dict, label_arrays = classify_internal_nodes(A, annotation_vector)
    selected_weights = {}
    for key in weight_dict.keys():
        selected_weights[key] = reduce_weights(A, weight_dict[key], label_arrays[key])
        important_kmers = [inversa(idx) for idx in selected_weights[key][1]]
