# kmeans com valor de K cumsum=0.9 FEITO
# cumsum=0.9 usar o aux de 0.9 para consturir arvore FEITO
# plotar pesos da regressão FEITO
# fazer arvore com alinhamento multiplo e neighbor-joining

"""
klearn.py

Pipeline for protein sequence classification and phylogenetics based on
k-mer composition.

Two independent (and combinable) modes:

  * Annotation-based classification (--anno FILE): trains a classifier per
    trait column in the annotation file and reports test accuracy, a
    confusion matrix and a ROC curve.

  * Tree-based node classification (--phylo --classify_nodes): builds a
    neighbor-joining tree from k-mer distances and, for every internal node
    (clade), fits a linear classifier that predicts clade membership from
    k-mer features -- no external annotation file needed, since the labels
    come from the tree topology itself.

At least one of the two modes must be requested. They can also be combined
in the same run.

Steps:
  1. Read protein sequences from a FASTA file.
  2. (optional) Read one or more binary trait annotations (tab-separated file).
  3. Build a k-mer composition feature matrix for every sequence.
  4. Reduce dimensionality with truncated SVD (PCA-style) and visualize it.
  5. Cluster samples with KMeans as a quick unsupervised sanity check.
  6. (annotation mode) Train/test a classifier for each trait and report
     accuracy, a confusion matrix and a ROC curve.
  7. (phylo mode) Build a neighbor-joining phylogenetic tree from k-mer
     distances, with or without bootstrap support.
  8. (tree-classification mode) Classify internal tree nodes (clades) from
     k-mer features and report the most discriminative k-mers for each node.
"""
import argparse
import os
from multiprocessing import Pool
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3D projection)

from Bio import SeqIO, Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from Bio.Phylo.Consensus import majority_consensus

from scipy.spatial.distance import pdist, squareform
from scipy.sparse import eye, bmat
from scipy.sparse.linalg import spsolve

from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc


ALPHABET = "ADQIMSYRCGLFTVNEHKPWX"  # last letter (X) is intentionally invalid/unused
VALID_AA_COUNT = 20  # only the first 20 letters of ALPHABET are valid amino acids
time_log = f"time_log_{os.getpid()}.txt"
start_time = time.time()

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Classification and phylogenetics of protein sequences from k-mer '
                     'composition. Two modes are available and can be combined: annotation-'
                     'based classification (--anno) and/or tree-based node classification '
                     '(--phylo --classify_nodes).'
    )
    parser.add_argument('--fasta', '-f', type=str, required=True,
                         help='FASTA file with protein sequences.')
    parser.add_argument('--anno', '-a', type=str, default=None,
                         help='Optional annotation file, tab-separated, first row as header '
                              'and first column as sequence ID. Each remaining column is a '
                              'trait and must be labeled as 0s or 1s. Multiple trait columns '
                              'are supported. If omitted, use --phylo with --classify_nodes '
                              'for annotation-free, tree-based node classification.')
    parser.add_argument('--phylo', action='store_true',
                         help='Build a phylogenetic tree from sequence k-mer distances.')
    parser.add_argument('--bootstraps', '-b', type=int, default=0,
                         help='Number of bootstrap replicates for the phylogenetic tree '
                              '(0 = build a single tree with no bootstrapping). Each '
                              'replicate resamples k-mer features (columns) with replacement, '
                              'keeping all samples/taxa, following the standard Felsenstein '
                              'bootstrap procedure.')
    parser.add_argument('--kmer', '-k', type=int, default=3,
                         help='k-mer size for feature table building.')
    parser.add_argument('--outdir', '-o', type=str, default='klearn_output',
                         help='Output directory.')
    parser.add_argument('--tree-cutoff', type=float, default=0.8,
                         help='Consensus cutoff for the bootstrap majority-rule tree.')
    parser.add_argument('--classify_nodes', action='store_true',
                         help='Classify internal tree nodes (clades) using k-mer features. '
                              'Requires --phylo. Does not require --anno.')
    parser.add_argument('--processes', '-p', type=int, default=1,
                         help='Number of processes to use for k-merization.')
    parser.add_argument('--classifier', choices=['linear', 'logreg'], default='logreg',
                         help='Classifier used for trait/node classification: "linear" is a '
                              'closed-form ridge/LS-SVM style solver, "logreg" is scikit-learn '
                              'LogisticRegression.')
    parser.add_argument('--test-size', type=float, default=0.25,
                         help='Fraction of samples held out to test classifier accuracy '
                              '(annotation mode only).')
    parser.add_argument('--assemble_kmers', action='store_true',
                         help='(Not yet implemented) Assemble discriminative k-mers back into '
                              'contigs.')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# k-mer <-> integer address encoding (generalized to arbitrary k)
# ---------------------------------------------------------------------------

def criapos(janela, k):
    """Map a k-length amino-acid window to a 1-based integer address in
    [1, VALID_AA_COUNT**k], or -1 if the window contains an invalid character
    or is not exactly length k."""
    if len(janela) != k:
        return -1
    pos = {aa: i for i, aa in enumerate(ALPHABET[:VALID_AA_COUNT])}
    idx = 0
    for ch in janela:
        p = pos.get(ch)
        if p is None:
            return -1
        idx = idx * VALID_AA_COUNT + p
    return idx + 1


def inversa(endereco, k):
    """Inverse of criapos: turn a 1-based integer address back into the
    k-length amino-acid string it represents."""
    idx = endereco - 1
    chars = []
    for _ in range(k):
        chars.append(ALPHABET[idx % VALID_AA_COUNT])
        idx //= VALID_AA_COUNT
    return ''.join(reversed(chars))


def kmerize_sequence(seq_str, k):
    """Build the k-mer composition vector for a single sequence string.
    Returns (vector, list_of_invalid_windows)."""
    seq_str = str(seq_str).upper()
    vector = np.zeros(VALID_AA_COUNT ** k)
    invalid = []
    for i in range(len(seq_str) - k + 1):
        window = seq_str[i:i + k]
        p = criapos(window, k)
        if p != -1:
            vector[p - 1] += 1
        else:
            invalid.append(window)
    return vector, invalid


# ---------------------------------------------------------------------------
# Sequence container
# ---------------------------------------------------------------------------

class seqObject:
    def __init__(self, id, seq):
        self.id = id
        self.seq = seq
        self.kmer_vector = None
        self.anno = None  # list of floats, one per trait; set by read_annotations

    def kmerize(self, k):
        vector, invalid = kmerize_sequence(self.seq, k)
        self.kmer_vector = vector
        for window in invalid:
            print(f"Warning: invalid character in kmer of {self.id}: {window}")


def read_fasta(fasta_file):
    try:
        sequences = []
        ids_seen = {}
        for record in SeqIO.parse(fasta_file, 'fasta'):
            seq_id = record.id
            if seq_id in ids_seen:
                ids_seen[seq_id] += 1
                new_id = f"{seq_id}_{ids_seen[seq_id]}"
                print(f"Warning: duplicate sequence ID '{seq_id}' in FASTA, "
                      f"renaming this occurrence to '{new_id}'.")
                seq_id = new_id
            else:
                ids_seen[seq_id] = 0
            sequences.append(seqObject(seq_id, str(record.seq)))
    except Exception as e:
        print("Fasta file error: " + str(e))
        return None
    if not sequences:
        print("Fasta file error: no sequences were parsed.")
        return None
    return sequences


# ---------------------------------------------------------------------------
# Annotation parsing (supports several trait columns)
# ---------------------------------------------------------------------------

def read_annotations(anno_file, sequences):
    """Read a tab-separated annotation file (first column = sequence ID,
    remaining columns = 0/1 trait labels) and attach `.anno` (list of floats)
    to every matching seqObject in `sequences`. Returns the list of trait
    (column) names."""
    df = pd.read_csv(anno_file, sep='\t', index_col=0, dtype=str)
    trait_names = list(df.columns)

    seq_by_id = {s.id: s for s in sequences}
    found_ids = set()

    for seq_id, row in df.iterrows():
        seq_id = str(seq_id)
        if seq_id not in seq_by_id:
            print(f"Warning: annotation for unknown sequence ID '{seq_id}' ignored.")
            continue
        try:
            values = [float(v) for v in row.tolist()]
        except (ValueError, TypeError):
            print(f"Warning: non-numeric annotation value for '{seq_id}', skipping.")
            continue
        seq_by_id[seq_id].anno = values
        found_ids.add(seq_id)

    missing = [s.id for s in sequences if s.id not in found_ids]
    if missing:
        preview = missing[:5]
        suffix = '...' if len(missing) > 5 else ''
        print(f"Warning: {len(missing)} sequence(s) had no annotation and will be dropped: "
              f"{preview}{suffix}")

    return trait_names


def create_annotation_vector(sequences):
    """Build an (n_samples, n_traits) matrix from each sequence's `.anno`."""
    n_traits = len(sequences[0].anno)
    matrix = np.zeros((len(sequences), n_traits))
    for i, seq in enumerate(sequences):
        matrix[i] = seq.anno
    return matrix


# ---------------------------------------------------------------------------
# Feature matrix / dimensionality reduction / clustering
# ---------------------------------------------------------------------------

def montaA(sequencias):
    n_features = sequencias[0].kmer_vector.shape[0]
    A = np.zeros((len(sequencias), n_features))
    for i, seq in enumerate(sequencias):
        A[i] = seq.kmer_vector
    return A


def create_sample_index(sequences):
    return {seq.id: seq for seq in sequences}


def singular(A, out_dir, labels=None, target_variance=0.70):
    """Truncated SVD dimensionality reduction. Returns:
       - A_pca_scaled: samples projected onto the top-3 components, scaled so
         that their captured variance ratio matches `target_variance`
         (used for 3D visualization).
       - A_k: samples projected onto the smallest number of components whose
         cumulative explained variance reaches `target_variance`.
    A 3D scatter plot of A_pca_scaled is saved to out_dir/pca_scaled.png."""
    max_components = max(1, min(A.shape[0], A.shape[1]) - 1)

    svd = TruncatedSVD(n_components=max_components, random_state=42)
    svd.fit(A)

    cumulative_variance = np.cumsum(svd.explained_variance_ratio_)
    reached = np.argmax(cumulative_variance >= target_variance)
    if cumulative_variance[reached] < target_variance:
        print(f"Warning: target variance {target_variance:.0%} not reached with "
              f"{max_components} components (max cumulative = "
              f"{cumulative_variance[-1]:.2%}). Using all available components.")
    optimal_k = min(reached + 1, max_components)

    svd_optimal = TruncatedSVD(n_components=optimal_k, random_state=42)
    A_k = svd_optimal.fit_transform(A)

    s = svd.singular_values_
    sum_s2_total = np.sum(s ** 2)
    n_top = min(3, max_components)

    if np.cumsum(svd.explained_variance_ratio_)[:n_top].sum() < 0.7:
        while np.cumsum(svd.explained_variance_ratio_)[:n_top].sum() < 0.7:
            optimal_visualization = optimal_k - 1
            svd_optimal = TruncatedSVD(n_components=optimal_visualization, random_state=42)
            A_k = svd_optimal.fit_transform(A)
            s = svd_optimal.singular_values_

    sum_s2_top = np.sum(s[:n_top] ** 2)
    alpha = np.sqrt((target_variance * sum_s2_total) / sum_s2_top) if sum_s2_top > 0 else 1.0

    V_top = svd.components_[:n_top, :]
    A_top = A @ V_top.T
    A_pca_scaled = A_top * alpha

    if n_top < 3:
        # pad with zeros so downstream 3D plotting code always has 3 columns
        A_pca_scaled = np.pad(A_pca_scaled, ((0, 0), (0, 3 - n_top)))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    if labels is not None:
        sc = ax.scatter(A_pca_scaled[:, 0], A_pca_scaled[:, 1], A_pca_scaled[:, 2],
                         c=labels, cmap='viridis')
    else:
        sc = ax.scatter(A_pca_scaled[:, 0], A_pca_scaled[:, 1], A_pca_scaled[:, 2])
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('PCA Scaled Sample Coordinates')
    if labels is not None:
        fig.colorbar(sc, ax=ax)
    plt.savefig(os.path.join(out_dir, "pca_scaled.png"))
    plt.close(fig)

    with open(os.path.join(out_dir, "svd_log.txt"), "w") as f:
        f.write(f"Optimal k: {optimal_k}\n")
        f.write(f"Cumulative variance: {cumulative_variance.tolist()}\n")
        f.write(f"A_k shape: {A_k.shape}\n")

    return A_pca_scaled, A_k, optimal_k


def run_kmeans(A_reduced, n_clusters, out_dir):
    """Fit KMeans on A_reduced (>=2 columns) and save a 2D scatter plot
    (first two components) colored by cluster assignment."""
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    model.fit(A_reduced)
    centroids = model.cluster_centers_

    plt.figure(figsize=(8, 6))
    plt.scatter(A_reduced[:, 0], A_reduced[:, 1], c=model.labels_, cmap='viridis', alpha=0.8)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, c='red', label='centroids')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'KMeans Clustering (k={n_clusters})')
    plt.legend()
    plt.savefig(os.path.join(out_dir, "kmeans.png"))
    plt.close()

    return model.labels_


# ---------------------------------------------------------------------------
# Linear classifiers (ridge/LS-SVM-style closed-form solver, and logistic
# regression), plus train/test accuracy evaluation with plots.
# ---------------------------------------------------------------------------

def resolve2(A, b):
    """Solve the dual (sample-space) formulation of ridge regression:
       [ I   A^T ] [x]   [0]
       [ A   -I  ] [y] = [b]
    which is equivalent to x = (A^T A + I)^-1 A^T b, but solved in an
    m-dimensional system (m = number of samples) instead of an n-dimensional
    one (n = number of features) -- useful when n >> m, as is typical for
    k-mer feature vectors."""
    m, n = A.shape

    In = eye(n, format="csr")
    Im = eye(m, format="csr")

    bm = np.concatenate([np.zeros(n), b])

    M = bmat([
        [In, A.T],
        [A, -Im]
    ], format="csr")

    x = spsolve(M, bm)
    return x[:n]


def fit_linear_classifier(A, indicadores, method='linear'):
    """Fit a linear classifier and return a per-feature weight vector `w`
    (larger |w_i| = more discriminative feature i)."""
    indicadores = np.asarray(indicadores)
    if method == 'linear':
        b = np.where(indicadores == 1, 12.0, -12.0)
        return resolve2(A, b)
    elif method == 'logreg':
        model = LogisticRegression(max_iter=1000)
        model.fit(A, indicadores)
        return model.coef_.ravel()
    else:
        raise ValueError(f"Unknown classifier method: {method}")

# plot the feature weights
def plot_weights(w, out_dir=None, trait_name='trait'):
    plt.figure()
    plt.bar(range(len(w)), w)
    plt.xlabel('Feature')
    plt.ylabel('Weight')
    plt.title(f'Feature weights for {trait_name}')
    plt.savefig(os.path.join(out_dir, f'{trait_name}_weights.png'))
    plt.close()


# kept for backwards compatibility with the original function name
def logistica(A, indicadores):
    return fit_linear_classifier(A, indicadores, method='linear')


def evaluate_classifier(A, indicadores, out_dir, trait_name='trait', method='logreg',
                         test_size=0.25, random_state=42):
    """Train/test split, fit `method` classifier, and save a figure with a
    confusion matrix and ROC curve to out_dir. Returns test accuracy."""
    indicadores = np.asarray(indicadores)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            A, indicadores, test_size=test_size, random_state=random_state,
            stratify=indicadores
        )
    except ValueError:
        # stratification can fail with very few samples per class
        X_train, X_test, y_train, y_test = train_test_split(
            A, indicadores, test_size=test_size, random_state=random_state
        )

    if method == 'linear':
        w = fit_linear_classifier(X_train, y_train, method='linear')
        scores_test = X_test @ w
        y_pred = np.where(scores_test >= 0, 1, 0)
    elif method == 'logreg':
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores_test = model.predict_proba(X_test)[:, 1]
    else:
        raise ValueError(f"Unknown classifier method: {method}")

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im = axes[0].imshow(cm, cmap='Blues')
    axes[0].set_title(f'{trait_name} — Confusion Matrix (acc={accuracy:.2f})')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    for (i, j), v in np.ndenumerate(cm):
        axes[0].text(j, i, str(v), ha='center', va='center')
    fig.colorbar(im, ax=axes[0])

    if len(np.unique(y_test)) > 1:
        fpr, tpr, _ = roc_curve(y_test, scores_test)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        axes[1].plot([0, 1], [0, 1], linestyle='--', color='gray')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title(f'{trait_name} — ROC Curve')
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'ROC unavailable\n(single class in test set)',
                      ha='center', va='center')
        axes[1].set_axis_off()

    plt.tight_layout()
    safe_name = "".join(c if c.isalnum() else "_" for c in trait_name)
    plt.savefig(os.path.join(out_dir, f"classifier_{safe_name}.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(out_dir, f"classifier_{safe_name}_report.txt"), "w") as f:
        f.write(f"Trait: {trait_name}\n")
        f.write(f"Classifier: {method}\n")
        f.write(f"Test accuracy: {accuracy:.4f}\n")
        f.write(f"Confusion matrix:\n{cm}\n")

    return accuracy


def reduce_weights(A, weight_vector, indicadores, classifier='linear', out_dir=None):
    """Pick the 10 most negative and 10 most positive weighted features,
    refit on just those 20 features, and return the refit weights together
    with their indices IN THE ORIGINAL FEATURE SPACE."""
    plot_weights(weight_vector, trait_name='all_weights', out_dir=out_dir)
    lowest_idx = np.argsort(weight_vector)[:10]
    highest_idx = np.argsort(weight_vector)[-10:][::-1]
    selected_idx = np.concatenate([lowest_idx, highest_idx])

    A_selected = A[:, selected_idx]
    weight_vector_selected = fit_linear_classifier(A_selected, indicadores, method=classifier)

    order = np.argsort(weight_vector_selected)
    lowest_local = order[:10]
    highest_local = order[-10:][::-1]
    final_local = np.concatenate([lowest_local, highest_local])

    final_original_idx = selected_idx[final_local]
    plot_weights(weight_vector_selected[final_local], trait_name='selected_weights', out_dir=out_dir)
    return weight_vector_selected[final_local], final_original_idx


# ---------------------------------------------------------------------------
# Phylogenetics
# ---------------------------------------------------------------------------

def neighbor_join_reduced(A, sample_labels, target_variance=0.90):
    max_components = max(1, min(A.shape[0], A.shape[1]) - 1)
    svd = TruncatedSVD(n_components=max_components, random_state=42)
    svd.fit(A)

    cumulative_variance = np.cumsum(svd.explained_variance_ratio_)
    optimal_k = min(np.argmax(cumulative_variance >= target_variance) + 1, max_components)

    svd_optimal = TruncatedSVD(n_components=optimal_k, random_state=42)
    A_k = svd_optimal.fit_transform(A)

    dist_condensed = pdist(A_k, metric='euclidean')
    square_form = squareform(dist_condensed)

    # Biopython's DistanceMatrix wants a lower-triangular "ragged" matrix:
    # row i must have exactly i+1 entries (including the 0 diagonal).
    lower_triangular = [list(square_form[i, :i + 1]) for i in range(square_form.shape[0])]

    bio_dm = DistanceMatrix(names=list(sample_labels), matrix=lower_triangular)
    constructor = DistanceTreeConstructor()
    tree = constructor.nj(bio_dm)
    return tree


def bootstrap(A, n_bootstrap, sequences, tree_cutoff, out_dir):
    """Standard (Felsenstein) non-parametric bootstrap: resample k-mer
    FEATURES (columns) with replacement while keeping every sample/taxon
    fixed, so the same set of unique tip labels is used to build every
    replicate tree. (Resampling samples/rows instead -- as an earlier
    version of this function did -- can produce duplicate tip labels within
    a single replicate, which Biopython's DistanceMatrix rejects with
    "ValueError: Duplicate names found".)"""
    trees = []
    sample_labels = [s.id for s in sequences]
    n_features = A.shape[1]

    for _ in range(n_bootstrap):
        feature_idx = np.random.choice(n_features, n_features, replace=True)
        A_bootstrap = A[:, feature_idx]
        tree = neighbor_join_reduced(A_bootstrap, sample_labels)
        trees.append(tree)

    consensus_tree = majority_consensus(trees, cutoff=tree_cutoff)
    consensus_tree.root.confidence = None

    fig, ax = plt.subplots(figsize=(10, 8))
    Phylo.draw(consensus_tree, axes=ax, do_show=False)
    plt.savefig(os.path.join(out_dir, "tree_image.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    Phylo.write(consensus_tree, os.path.join(out_dir, "majority_consensus_tree.nwk"), "newick")

    return consensus_tree


def classify_internal_nodes(tree, A, sequences, classifier='linear'):
    """For every internal node (clade) of `tree`, build a 0/1 indicator
    vector over samples (1 = descendant of that clade) and fit a linear
    classifier against the k-mer feature matrix `A`. Returns
    (weight_dict, label_dict) both keyed by clade object."""
    id_to_row = {seq.id: i for i, seq in enumerate(sequences)}

    labels = {}
    for node in tree.get_nonterminals():
        descendant_ids = {clade.name for clade in node.get_terminals()}
        vector = np.zeros(A.shape[0])
        for seq_id, row in id_to_row.items():
            if seq_id in descendant_ids:
                vector[row] = 1.0
        labels[node] = vector

    weight_dict = {
        node: fit_linear_classifier(A, vector, method=classifier)
        for node, vector in labels.items()
    }

    return weight_dict, labels


# ---------------------------------------------------------------------------
# (planned) k-mer assembly -- not yet implemented
# ---------------------------------------------------------------------------

def assemble_kmers(important_kmers, sequences, k):
    """Placeholder for reassembling the most discriminative k-mers back into
    longer contigs (e.g. via a De Bruijn graph over `important_kmers`,
    walking overlaps of length k-1). Not implemented yet."""
    raise NotImplementedError(
        "--assemble_kmers is not implemented yet. It is planned to build a "
        "De Bruijn-style graph over the discriminative k-mers found by "
        "--classify_nodes and greedily walk it into contigs."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    time_nodes = 0
    time_tree = 0
    time_kmerization = 0
    time_visualization = 0
    args = parse_args()

    annotation_mode = args.anno is not None
    tree_mode = args.phylo

    if not annotation_mode and not tree_mode:
        print("Error: nothing to do. Provide --anno for annotation-based classification, "
              "and/or --phylo (with --classify_nodes) for tree-based node classification.")
        return

    if args.classify_nodes and not args.phylo:
        print("Error: --classify_nodes requires --phylo to build a tree first.")
        return

    out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)

    k = args.kmer
    processes = args.processes
    classifier_method = args.classifier
    test_size = args.test_size

    # 1. Read sequences
    sequences = read_fasta(args.fasta)
    if not sequences:
        print("No sequences found. Exiting.")
        return

    # 2. Read annotations, if provided (annotation-based classification mode).
    #    Tree-based node classification does not need this at all.
    trait_names = []
    annotation_matrix = None
    if annotation_mode:
        trait_names = read_annotations(args.anno, sequences)
        sequences = [s for s in sequences if s.anno is not None]
        if not sequences:
            print("No sequences with matching annotations found. Exiting.")
            return
        if not trait_names:
            print("No trait columns found in annotation file. Exiting.")
            return
        annotation_matrix = create_annotation_vector(sequences)

    # 3. K-merize (parallel or sequential)
    kmerization_time = time.time()

    if k > 4:
        print(f"Note: k={k} means a {VALID_AA_COUNT**k:,}-dimensional feature vector per "
              f"sequence; this may use a lot of memory.")

    if processes > 1:
        with Pool(processes=processes) as pool:
            results = pool.starmap(kmerize_sequence, [(s.seq, k) for s in sequences])
    else:
        results = [kmerize_sequence(s.seq, k) for s in sequences]

    for seq, (vector, invalid) in zip(sequences, results):
        seq.kmer_vector = vector
        for window in invalid:
            print(f"Warning: invalid character in kmer of {seq.id}: {window}")
    kmerization_time = time.time() - kmerization_time
    print(f"K-merization time: {kmerization_time:.2f} seconds")

    # 4. Build feature matrix
    A = montaA(sequences)
    np.save(os.path.join(out_dir, "feature_matrix.npy"), A)

    # 5. SVD / PCA visualization
    visualization_time = time.time()
    color_labels = annotation_matrix[:, 0] if annotation_matrix is not None else None
    A_reduced, A_k, optimal_k = singular(A, out_dir, labels=color_labels)
    np.save(os.path.join(out_dir, "A_reduced.npy"), A_reduced)
    np.save(os.path.join(out_dir, "A_k.npy"), A_k)

    # 6. Unsupervised clustering as a visual sanity check
    run_kmeans(A_reduced, optimal_k, out_dir)
    visualization_time = time.time() - visualization_time
    print(f"Visualization time: {visualization_time:.2f} seconds")

    # 7. Annotation-based classification accuracy for every trait
    if annotation_mode:
        for t_idx, trait_name in enumerate(trait_names):
            y = annotation_matrix[:, t_idx]
            if len(np.unique(y)) < 2:
                print(f"Skipping trait '{trait_name}': only one class present.")
                continue
            acc = evaluate_classifier(A, y, out_dir, trait_name=trait_name,
                                       method=classifier_method, test_size=test_size)
            print(f"Trait '{trait_name}': test accuracy = {acc:.4f}")

    # 8. Phylogenetics
    tree = None
    if tree_mode:
        tree_time = time.time()
        sample_labels = [seq.id for seq in sequences]
        if args.bootstraps == 0:
            tree = neighbor_join_reduced(A, sample_labels)
            Phylo.write(tree, os.path.join(out_dir, "tree.newick"), "newick")
            fig, ax = plt.subplots(figsize=(10, 8))
            Phylo.draw(tree, axes=ax, do_show=False)
            plt.savefig(os.path.join(out_dir, "tree_image.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            tree = bootstrap(A, args.bootstraps, sequences, args.tree_cutoff, out_dir)
        tree_time = time.time() - tree_time
        print(f"Phylogenetic tree construction time: {tree_time:.2f} seconds")

    # 9. Tree-based internal node classification (annotation-free)
    if args.classify_nodes:
        time_nodes = time.time()
        weight_dict, label_arrays = classify_internal_nodes(
            tree, A, sequences, classifier=classifier_method
        )
        report_path = os.path.join(out_dir, "node_classification.txt")
        with open(report_path, "w") as f:
            for node, w in weight_dict.items():
                indicadores = label_arrays[node]
                _, selected_idx = reduce_weights(A, w, indicadores, classifier=classifier_method, out_dir=out_dir)
                important_kmers = [inversa(int(idx) + 1, k) for idx in selected_idx]
                node_label = node.name if node.name else f"node_{id(node)}"
                f.write(f"{node_label}\t{','.join(important_kmers)}\n")
        time_nodes = time.time() - time_nodes
        print(f"Internal node classification time: {time_nodes:.2f} seconds")
        print(f"Internal node classification written to {report_path}")

    # 10. k-mer assembly (not yet implemented)
    if args.assemble_kmers:
        print("--assemble_kmers was requested but this feature is not implemented yet; skipping.")

    print(f"Done. Results written to {out_dir}")

    with open(time_log, "a") as f:
        f.write(f"total time\t{time.time() - start_time}\n")
        f.write(f"k-merization time\t{kmerization_time:.2f} seconds\n")
        f.write(f"visualization time\t{visualization_time:.2f} seconds\n")
        f.write(f"tree time\t{tree_time:.2f} seconds\n")
        f.write(f"node classification time\t{time_nodes:.2f} seconds\n")


if __name__ == '__main__':
    main()
