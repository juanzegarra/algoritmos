# Paralelismo no bootstrap das árvores FEITO
# Fazer plot do P(x) ordenado cor diferente para os labels
# Retornar kmers dos labels FEITO
# Kmer = 4
# kmeans com valor de K cumsum=0.9 FEITO
# cumsum=0.9 usar o aux de 0.9 para consturir arvore FEITO
# plotar pesos da regressão FEITO
# fazer arvore com alinhamento multiplo e neighbor-joining FEITO
# Implementar paralelismo no classificador dos nós
# Paralelismo no bootstrap das árvores FEITO
# Fazer plot do P(x) ordenado cor diferente para os labels
# Aumentar numero de kmers para classificação FEITO (adicionei flag para dar entrada no número de kmers desejados)
# Retornar kmers dos labels FEITO
# Kmer = 4
# Comparar filogenia com filogenia classica
# Tentar colocar como esparsa a matriz de features ao inves de inicializar como zero 
# Cortar os features não importantes para fazer bootstrapping, não selecionar da matriz completa de features. 
# Criar função para selecionar threasholds
# fazer montagem 
# tentar achar estrutura 
# Tentar debuggar quitapleta 

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
import multiprocessing as mp
import time
from collections import defaultdict


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
import scipy.io

from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc


ALPHABET = "ADQIMSYRCGLFTVNEHKPWX"  # last letter (X) is intentionally invalid/unused
VALID_AA_COUNT = 20  # only the first 20 letters of ALPHABET are valid amino acids
DEFAULT_N_WEIGHTS = 20  # top/bottom N features kept per side when reducing weights

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
                              'for annotation-free, tree-based node classification.'
                              'Column with name Display is used for labelling figures in output plots.')
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
                         help='Consensus cutoff for the bootstrap majority-rule tree. (default = 0.8)')
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
    parser.add_argument('--weight_reduction', type=int, default=None,
                         help='Reduce weights by selecting the N top and N lowest most discriminative k-mers.')
    parser.add_argument('--assemble_kmers', action='store_true',
                         help='(Not yet implemented) Assemble discriminative k-mers back into '
                              'contigs.')

    return parser.parse_args()

# ---------------------------------------------------------------------------
#  Global variables for shared multiprocessing memory
# ---------------------------------------------------------------------------

global_A = None
global_labels = None
display = None

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
        self.display_name = None

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
    and `.display_name` to every matching seqObject in `sequences`."""
    df = pd.read_csv(anno_file, sep='\t', index_col=0, dtype=str)
    trait_names = list(df.columns)

    # Isolate and ignore Display column for trait training
    has_display = "Display" in trait_names
    if has_display:
        trait_names.remove("Display")

    seq_by_id = {s.id: s for s in sequences}
    found_ids = set()

    for seq_id, row in df.iterrows():
        seq_id = str(seq_id)
        if seq_id not in seq_by_id:
            print(f"Warning: annotation for unknown sequence ID '{seq_id}' ignored.")
            continue

        if has_display and pd.notna(row.get("Display")):
            seq_by_id[seq_id].display_name = str(row["Display"])

        try:
            # Extract float values only for actual traits, ignoring 'Display'
            values = [float(row[col]) for col in trait_names]
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
    explained_variance = np.sort(svd.explained_variance_ratio_)

    scree_fig, scree_ax = plt.subplots(figsize=(10, 6))
    scree_ax.plot(explained_variance, 'o-')
    scree_ax.set_xlabel('Singular Value Index')
    scree_ax.set_ylabel('Singular Value')
    scree_ax.set_title('Singular Values')
    scree_fig.savefig(os.path.join(out_dir, "svd.png"))
    plt.close(scree_fig)

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
def plot_weights(w, out_dir, trait_name='trait'):
    """Scatter plot of a classifier's per-feature weights, sorted ascending.
    Saved to out_dir/<trait_name>_weights.png -- pass a distinct
    `trait_name` per call (e.g. via reduce_weights' `label`), or repeated
    calls will overwrite each other's plot."""
    w_sorted = np.sort(w)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(range(len(w_sorted)), w_sorted, marker='*', s=10)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Feature (sorted by weight)')
    ax.set_ylabel('Weight')
    ax.set_title(f'Feature weights — {trait_name}')
    safe_name = "".join(c if c.isalnum() else "_" for c in trait_name)
    fig.savefig(os.path.join(out_dir, f'{safe_name}_weights.png'))
    plt.close(fig)


# kept for backwards compatibility with the original function name
def logistica(A, indicadores):
    return fit_linear_classifier(A, indicadores, method='linear')

## Change this function to accept weight reduction and refitting
def evaluate_classifier(A, indicadores, out_dir, trait_name='trait', method='logreg',
                         test_size=0.25, random_state=42, sample_ids=None,
                         weight_reduction=None, k=None):
    indicadores = np.asarray(indicadores)
    idx_all = np.arange(A.shape[0])

    try:
        idx_train, idx_test = train_test_split(
            idx_all, test_size=test_size, random_state=random_state,
            stratify=indicadores
        )
    except ValueError:
        idx_train, idx_test = train_test_split(
            idx_all, test_size=test_size, random_state=random_state
        )

    X_train, X_test = A[idx_train], A[idx_test]
    y_train, y_test = indicadores[idx_train], indicadores[idx_test]

    ids_test = ([sample_ids[i] for i in idx_test] if sample_ids is not None
                else [str(i) for i in idx_test])
    ids_all = sample_ids if sample_ids is not None else [str(i) for i in range(A.shape[0])]

    selected_idx = None

    # Helper to convert raw logits to probabilities [0, 1]
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    if method == 'linear' and weight_reduction is not None:
        w = fit_linear_classifier(X_train, y_train, method='linear')
        w_selected, selected_idx = reduce_weights(
            X_train, w, y_train, classifier='linear', out_dir=out_dir,
            n_weights=weight_reduction, label=trait_name
        )

        # Test set predictions
        scores_test = sigmoid(X_test[:, selected_idx] @ w_selected)
        y_pred_test = np.where(scores_test >= 0.5, 1, 0)

        # All data predictions
        scores_all = sigmoid(A[:, selected_idx] @ w_selected)
        y_pred_all = np.where(scores_all >= 0.5, 1, 0)

    elif method == 'linear':
        w = fit_linear_classifier(X_train, y_train, method='linear')

        # Test set predictions
        scores_test = sigmoid(X_test @ w)
        y_pred_test = np.where(scores_test >= 0.5, 1, 0)

        # All data predictions
        scores_all = sigmoid(A @ w)
        y_pred_all = np.where(scores_all >= 0.5, 1, 0)

    elif method == 'logreg':
        if weight_reduction is not None:
            print(f"Note: --weight_reduction only applies to --classifier linear; "
                  f"ignoring it for trait '{trait_name}' (classifier=logreg).")
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Test set predictions
        y_pred_test = model.predict(X_test)
        scores_test = model.predict_proba(X_test)[:, 1]

        # All data predictions
        y_pred_all = model.predict(A)
        scores_all = model.predict_proba(A)[:, 1]
    else:
        raise ValueError(f"Unknown classifier method: {method}")

    accuracy = accuracy_score(y_test, y_pred_test)
    cm = confusion_matrix(y_test, y_pred_test)

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
        if selected_idx is not None:
            if k is not None:
                important_kmers = [inversa(int(idx) + 1, k) for idx in selected_idx]
                f.write(f"Selected k-mers ({len(important_kmers)}): "
                        f"{', '.join(important_kmers)}\n")
            else:
                f.write(f"Selected feature indices ({len(selected_idx)}): "
                        f"{selected_idx.tolist()} (pass k-mer size to decode as sequences)\n")

    # Generate two plots: one strictly for the test partition, one for the whole dataset
    plot_logit_classification(ids_test, y_test, scores_test, y_pred_test, out_dir, f"{trait_name}_test")
    plot_logit_classification(ids_all, indicadores, scores_all, y_pred_all, out_dir, f"{trait_name}_all")

    return accuracy

def plot_logit_classification(sample_ids, y_true, scores, y_pred, out_dir, trait_name):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = np.asarray(scores)
    sample_ids = list(sample_ids)

    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_true = y_true[order]
    sorted_pred = y_pred[order]
    sorted_ids = [sample_ids[i] for i in order]
    correct = sorted_true == sorted_pred

    fig_height = max(4, len(sorted_scores) * 0.28)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    y_positions = np.arange(len(sorted_scores))

    classes = sorted(set(y_true.tolist()))
    palette = {classes[0]: 'tab:blue', classes[-1]: 'tab:orange'} if len(classes) > 1 \
        else {classes[0]: 'tab:blue'}

    for cls, color in palette.items():
        mask = sorted_true == cls
        ax.scatter(sorted_scores[mask & correct], y_positions[mask & correct],
                   color=color, marker='o', s=45,
                   label=f'true label {int(cls)} — correctly classified')
        ax.scatter(sorted_scores[mask & ~correct], y_positions[mask & ~correct],
                   color=color, marker='x', linewidths=2,
                   s=70, label=f'true label {int(cls)} — misclassified')

    # Threshold updated to 0.5 for probability bounds
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=1, label='decision threshold (0.5)')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(sorted_ids, fontsize=7)
    ax.set_xlabel('Predicted Probability')
    ax.set_title(f'{trait_name} — sequence classification vs. true label')
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()

    safe_name = "".join(c if c.isalnum() else "_" for c in trait_name)
    fig_path = os.path.join(out_dir, f"probability_classification_{safe_name}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)

    table_path = os.path.join(out_dir, f"probability_classification_{safe_name}.tsv")
    with open(table_path, "w") as f:
        f.write("sequence_id\ttrue_label\tpredicted_label\tprobability\n")
        for sid, t, p, s in zip(sorted_ids, sorted_true, sorted_pred, sorted_scores):
            f.write(f"{sid}\t{int(t)}\t{int(p)}\t{s:.4f}\n")

    return fig_path, table_path

def reduce_weights(A, weight_vector, indicadores, classifier='linear', out_dir=None,
                    n_weights=DEFAULT_N_WEIGHTS, label='weights'):
    """Pick the `n_weights` most negative and `n_weights` most positive
    weighted features, refit on just those 2*n_weights features, and return
    the refit weights together with their indices IN THE ORIGINAL FEATURE
    SPACE. If `out_dir` is given, also saves two weight plots (before and
    after the refit) named from `label` -- pass a distinct string `label`
    per call (e.g. the trait name or a tree node's name), or repeated calls
    will overwrite each other's plots."""
    if out_dir is not None:
        plot_weights(weight_vector, out_dir, trait_name=f'{label}_all')

    lowest_idx = np.argsort(weight_vector)[:n_weights]
    highest_idx = np.argsort(weight_vector)[-n_weights:][::-1]
    selected_idx = np.concatenate([lowest_idx, highest_idx])

    A_selected = A[:, selected_idx]
    weight_vector_selected = fit_linear_classifier(A_selected, indicadores, method=classifier)

    order = np.argsort(weight_vector_selected)
    lowest_local = order[:n_weights]
    highest_local = order[-n_weights:][::-1]
    final_local = np.concatenate([lowest_local, highest_local])

    final_original_idx = selected_idx[final_local]
    if out_dir is not None:
        plot_weights(weight_vector_selected[final_local], out_dir, trait_name=f'{label}_selected')
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

def initialize_bootstrap(A, sample_labels):
    global global_A
    global global_labels
    global_A = A
    global_labels = sample_labels

def run_single_bootstrap(seed):
    """Runs inside a worker process. Reads the feature matrix and sample
    labels from the globals set by `initialize_bootstrap` (via the Pool
    initializer) instead of receiving them as arguments, so only the small
    `seed` integer needs to be pickled per task -- not the whole matrix."""
    global global_A, global_labels
    rng = np.random.RandomState(seed)
    n_features = global_A.shape[1]
    feature_idx = rng.choice(n_features, n_features, replace=True)
    A_bootstrap = global_A[:, feature_idx]
    return neighbor_join_reduced(A_bootstrap, global_labels)


def bootstrap(A, n_bootstrap, sequences, tree_cutoff, out_dir, processes=None):
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

    if not processes or processes <= 1 or n_bootstrap <= 1:
        for _ in range(n_bootstrap):
            feature_idx = np.random.choice(n_features, n_features, replace=True)
            A_bootstrap = A[:, feature_idx]
            tree = neighbor_join_reduced(A_bootstrap, sample_labels)
            trees.append(tree)
    else:
        n_workers = min(processes, mp.cpu_count(), n_bootstrap)
        seeds = np.random.randint(0, 2 ** 32 - 1, size=n_bootstrap).tolist()

        # A and sample_labels are sent to each worker ONCE via the Pool
        # initializer; each task then only pickles its small seed integer
        # (see run_single_bootstrap, which reads them back from the globals
        # the initializer sets INSIDE each worker process).
        with mp.Pool(
            processes=n_workers,
            initializer=initialize_bootstrap,
            initargs=(A, sample_labels),
        ) as pool:
            trees = pool.map(run_single_bootstrap, seeds)

    consensus_tree = majority_consensus(trees, cutoff=tree_cutoff)
    consensus_tree.root.confidence = None

    fig, ax = plt.subplots(figsize=(10, 8))
    Phylo.draw(consensus_tree, axes=ax, do_show=False)
    plt.savefig(os.path.join(out_dir, "tree_image.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    Phylo.write(consensus_tree, os.path.join(out_dir, "majority_consensus_tree.nwk"), "newick")

    return consensus_tree

def init_classifier_worker(A, classifier_method):
    global classifier_A, classifier_method_global
    classifier_A = A
    classifier_method_global = classifier_method


def fit_node_classifier_worker(vector):
    """Runs inside a worker process; reads A and the classifier method from
    the globals set by init_classifier_worker (via the Pool initializer)."""
    global classifier_A, classifier_method_global
    return fit_linear_classifier(classifier_A, vector, method=classifier_method_global)


def classify_internal_nodes(tree, A, sequences, classifier='linear', processes=1):
    """For every internal node (clade) of `tree`, build a 0/1 indicator
    vector over samples (1 = descendant of that clade) and fit a linear
    classifier against the k-mer feature matrix `A`. Returns
    (weight_dict, label_dict) both keyed by clade object.

    If `processes` > 1, one classifier is fit per node in parallel (the
    feature matrix `A` is sent to each worker once via the Pool
    initializer, rather than once per node)."""
    # Map rows to display names (or fallback to ID) to match tree tips
    id_to_row = {seq.display_name if seq.display_name else seq.id: i for i, seq in enumerate(sequences)}

    labels = {}
    for node in tree.get_nonterminals():
        descendant_ids = {clade.name for clade in node.get_terminals()}
        vector = np.zeros(A.shape[0])
        for seq_id, row in id_to_row.items():
            if seq_id in descendant_ids:
                vector[row] = 1.0
        labels[node] = vector

    nodes = list(labels.keys())
    vectors = [labels[node] for node in nodes]

    if processes and processes > 1 and len(nodes) > 1:
        n_workers = min(processes, mp.cpu_count(), len(nodes))
        with mp.Pool(n_workers, initializer=init_classifier_worker,
                     initargs=(A, classifier)) as pool:
            weight_list = pool.map(fit_node_classifier_worker, vectors)
    else:
        weight_list = [fit_linear_classifier(A, vector, method=classifier) for vector in vectors]

    weight_dict = dict(zip(nodes, weight_list))

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
    start_time = time.time()
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
    time_log_path = os.path.join(out_dir, f"time_log_{os.getpid()}.txt")

    k = args.kmer
    processes = args.processes
    classifier_method = args.classifier
    test_size = args.test_size
    weight_reduction = args.weight_reduction

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
    time_kmerization = time.time()

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
    time_kmerization = time.time() - time_kmerization
    print(f"K-merization time: {time_kmerization:.2f} seconds")

    # 4. Build feature matrix
    A = montaA(sequences)
    scipy.io.savemat(os.path.join(out_dir, "feature_matrix.mat"), {"A": A})
    np.save(os.path.join(out_dir, "feature_matrix.npy"), A)

    # 5. SVD / PCA visualization
    time_visualization = time.time()
    color_labels = annotation_matrix[:, 0] if annotation_matrix is not None else None
    A_reduced, A_k, optimal_k = singular(A, out_dir, labels=color_labels)
    np.save(os.path.join(out_dir, "A_reduced.npy"), A_reduced)
    np.save(os.path.join(out_dir, "A_k.npy"), A_k)

    # 6. Unsupervised clustering as a visual sanity check
    run_kmeans(A_reduced, optimal_k, out_dir)
    time_visualization = time.time() - time_visualization
    print(f"Visualization time: {time_visualization:.2f} seconds")

    # 7. Annotation-based classification accuracy for every trait
    if annotation_mode:
        display_ids = [s.display_name if s.display_name else s.id for s in sequences]
        for t_idx, trait_name in enumerate(trait_names):
            y = annotation_matrix[:, t_idx]
            if len(np.unique(y)) < 2:
                print(f"Skipping trait '{trait_name}': only one class present.")
                continue
            acc = evaluate_classifier(A, y, out_dir, trait_name=trait_name,
                                       method=classifier_method, test_size=test_size,
                                       sample_ids=display_ids,
                                       weight_reduction=weight_reduction, k=k)
            print(f"Trait '{trait_name}': test accuracy = {acc:.4f}")

    # 8. Phylogenetics
    tree = None
    if tree_mode:
        time_tree = time.time()
        sample_labels = [seq.display for seq in sequences, else seq.id for seq in sequences]
        if args.bootstraps == 0:
            tree = neighbor_join_reduced(A, sample_labels)
            Phylo.write(tree, os.path.join(out_dir, "tree.newick"), "newick")
            fig, ax = plt.subplots(figsize=(10, 8))
            Phylo.draw(tree, axes=ax, do_show=False)
            plt.savefig(os.path.join(out_dir, "tree_image.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            tree = bootstrap(A, args.bootstraps, sequences, args.tree_cutoff, out_dir, processes=processes)
        time_tree = time.time() - time_tree
        print(f"Phylogenetic tree construction time: {time_tree:.2f} seconds")

    # 9. Tree-based internal node classification (annotation-free)
    if args.classify_nodes:
        time_nodes = time.time()
        weight_dict, label_arrays = classify_internal_nodes(
            tree, A, sequences, classifier=classifier_method, processes=processes
        )
        report_path = os.path.join(out_dir, "node_classification.txt")
        with open(report_path, "w") as f:
            for node, w in weight_dict.items():
                indicadores = label_arrays[node]
                node_label = node.name if node.name else f"node_{id(node)}"
                _, selected_idx = reduce_weights(
                    A, w, indicadores, classifier=classifier_method, out_dir=out_dir,
                    n_weights=weight_reduction or DEFAULT_N_WEIGHTS, label=node_label
                )
                important_kmers = [inversa(int(idx) + 1, k) for idx in selected_idx]
                f.write(f"{node_label}\t{','.join(important_kmers)}\n")
        time_nodes = time.time() - time_nodes
        print(f"Internal node classification time: {time_nodes:.2f} seconds")
        print(f"Internal node classification written to {report_path}")

    # 10. k-mer assembly (not yet implemented)
    if args.assemble_kmers:
        print("--assemble_kmers was requested but this feature is not implemented yet; skipping.")

    print(f"Done. Results written to {out_dir}")

    with open(time_log_path, "a") as f:
        f.write(f"total time\t{time.time() - start_time}\n")
        f.write(f"k-merization time\t{time_kmerization:.2f} seconds\n")
        f.write(f"visualization time\t{time_visualization:.2f} seconds\n")
        if tree_mode:
            f.write(f"tree time\t{time_tree:.2f} seconds\n")
        f.write(f"node classification time\t{time_nodes:.2f} seconds\n")


if __name__ == '__main__':
    main()
