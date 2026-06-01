"""
Visualize a trained min-char-rnn after training is done.

Loads the saved model from `model.npz`, runs a forward pass over the first
`--length` characters of `input.txt`, and plots:

  1) hidden_states_heatmap.png
       Heatmap of every hidden unit's activation at every timestep.
       Y-axis: hidden units (h0, h1, ...). X-axis: input character.
       Works for any `hidden_size`.

  2) output_probabilities.png
       Heatmap of the model's next-char probability distribution at every
       position, with the actual next character marked.

  3) hidden_states_trajectory.png   (only when hidden_size == 2)
       2D scatter of every hidden state visited, colored by the input
       character that produced it, with grey arrows showing the temporal
       trajectory through state space.

  4) hidden_states_by_target.png    (only when hidden_size == 2)
       Same scatter, colored by the *next* (target) character.

  5) learning_curve.png
       Per-window training loss vs iteration (from model.npz).

  6) hidden_states_pca_context_labels.png
       2D PCA of hidden states; annotation = prev2 + current char.

  7) hidden_states_mds_context_labels.png
       2D MDS from the same euclidean distances as the clustermap dendrogram.

  8) hidden_states_clustermap.png
       Heatmap of timesteps × hidden units with row/column dendrograms
       (average linkage). Row labels: two preceding chars + current char.

  9) weights.png
       Side-by-side heatmaps of final input weights (char columns × hidden rows)
       and recurrent hidden→hidden weights (h0..h{n-1} in index order).

Usage:
    python visualize.py                  # default: 50 chars
    python visualize.py --length 80
    python visualize.py --length 50 --out-dir plots
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist, squareform


def load_model(path: str = "model.npz"):
    data = np.load(path, allow_pickle=False)
    model = {
        "weights_input_to_hidden":  data["weights_input_to_hidden"],
        "weights_hidden_to_hidden": data["weights_hidden_to_hidden"],
        "weights_hidden_to_output": data["weights_hidden_to_output"],
        "bias_hidden":              data["bias_hidden"],
        "bias_output":              data["bias_output"],
        "chars":                    [str(c) for c in data["chars"]],
        "hidden_size":              int(data["hidden_size"]),
        "vocab_size":               int(data["vocab_size"]),
    }
    if "loss_iterations" in data.files:
        model["loss_iterations"] = data["loss_iterations"]
        model["loss_smooth"] = data["loss_smooth"]
        model["loss_window"] = data["loss_window"]
    return model


def forward_pass(model, text: str):
    """Run the trained RNN over `text` and return per-timestep states + probs."""
    hidden_size = model["hidden_size"]
    vocab_size  = model["vocab_size"]
    chars       = model["chars"]
    char_to_index = {c: i for i, c in enumerate(chars)}

    weights_input_to_hidden  = model["weights_input_to_hidden"]
    weights_hidden_to_hidden = model["weights_hidden_to_hidden"]
    weights_hidden_to_output = model["weights_hidden_to_output"]
    bias_hidden              = model["bias_hidden"]
    bias_output              = model["bias_output"]

    hidden_state = np.zeros((hidden_size, 1))
    hidden_states = np.zeros((len(text), hidden_size))
    output_probs  = np.zeros((len(text), vocab_size))

    for t, char in enumerate(text):
        input_one_hot = np.zeros((vocab_size, 1))
        input_one_hot[char_to_index[char]] = 1
        hidden_state = np.tanh(
            weights_input_to_hidden  @ input_one_hot +
            weights_hidden_to_hidden @ hidden_state  +
            bias_hidden
        )
        logits = weights_hidden_to_output @ hidden_state + bias_output
        exp = np.exp(logits - np.max(logits))
        probs = exp / np.sum(exp)

        hidden_states[t] = hidden_state.ravel()
        output_probs[t]  = probs.ravel()

    return hidden_states, output_probs


def plot_state_trajectory(hidden_states, color_by_chars, chars, title, save_path):
    """2D scatter of hidden states colored by some categorical char per timestep."""
    if hidden_states.shape[1] != 2:
        raise ValueError(
            f"This plot expects hidden_size == 2, got {hidden_states.shape[1]}. "
            f"Re-train with hidden_size = 2 (already the default in min-char-rnn.py)."
        )

    cmap = plt.get_cmap("tab10")
    char_to_color = {c: cmap(i) for i, c in enumerate(chars)}

    fig, ax = plt.subplots(figsize=(8, 7))

    xs, ys = hidden_states[:, 0], hidden_states[:, 1]
    ax.plot(xs, ys, color="lightgrey", linewidth=0.5, zorder=1)
    ax.quiver(
        xs[:-1], ys[:-1],
        xs[1:] - xs[:-1], ys[1:] - ys[:-1],
        angles="xy", scale_units="xy", scale=1,
        color="lightgrey", width=0.002, headwidth=4, alpha=0.6, zorder=1,
    )

    for c in chars:
        mask = np.array([ch == c for ch in color_by_chars])
        if not mask.any():
            continue
        ax.scatter(
            xs[mask], ys[mask],
            color=char_to_color[c], label=repr(c), s=30,
            edgecolor="black", linewidth=0.3, zorder=3,
        )

    ax.set_xlabel("hidden unit 0")
    ax.set_ylabel("hidden unit 1")
    ax.set_title(title)
    ax.legend(title="char", loc="best", framealpha=0.9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_hidden_states_heatmap(text, hidden_states, save_path):
    """Heatmap of every hidden unit's tanh activation over the sequence.

    rows = hidden units, columns = timesteps, color = activation in [-1, 1].
    """
    length, hidden_size = hidden_states.shape

    fig, ax = plt.subplots(figsize=(max(12, length * 0.15),
                                    max(2.5, hidden_size * 0.35)))
    im = ax.imshow(
        hidden_states.T,
        aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
        interpolation="nearest", origin="lower",
    )

    ax.set_yticks(range(hidden_size))
    ax.set_yticklabels([f"h{i}" for i in range(hidden_size)])
    ax.set_xticks(range(length))
    ax.set_xticklabels(list(text), fontsize=7)
    ax.set_xlabel("timestep / input character")
    ax.set_ylabel("hidden unit")
    ax.set_title("Hidden state activations (tanh output) over the input sequence")

    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="activation (tanh)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"wrote {save_path}")


def average_linkage_hierarchy(rows):
    """Average-linkage clustering; returns (linkage, leaf_order).

    linkage has shape (n-1, 4) with columns [left_id, right_id, distance, count]
    in the same convention as scipy.cluster.hierarchy.linkage.
    """
    n_rows = rows.shape[0]
    if n_rows == 0:
        return np.zeros((0, 4)), []
    if n_rows == 1:
        return np.zeros((0, 4)), [0]

    distances = np.linalg.norm(rows[:, None, :] - rows[None, :, :], axis=2)
    clusters = [
        {"indices": [i], "members": [i], "cluster_id": i, "size": 1}
        for i in range(n_rows)
    ]
    linkage = []
    next_cluster_id = n_rows

    while len(clusters) > 1:
        best_pair = None
        best_distance = np.inf

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                member_distances = distances[np.ix_(
                    clusters[i]["members"],
                    clusters[j]["members"],
                )]
                distance = float(np.mean(member_distances))
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (i, j)

        left, right = best_pair
        left_cluster, right_cluster = clusters[left], clusters[right]
        linkage.append([
            left_cluster["cluster_id"],
            right_cluster["cluster_id"],
            best_distance,
            left_cluster["size"] + right_cluster["size"],
        ])
        merged = {
            "indices": left_cluster["indices"] + right_cluster["indices"],
            "members": left_cluster["members"] + right_cluster["members"],
            "cluster_id": next_cluster_id,
            "size": left_cluster["size"] + right_cluster["size"],
        }
        next_cluster_id += 1
        clusters[left] = merged
        del clusters[right]

    return np.array(linkage), clusters[0]["indices"]


def average_linkage_cluster_order(rows):
    """Return row indices ordered by a small average-linkage clustering pass."""
    _, order = average_linkage_hierarchy(rows)
    return order


def display_char(char):
    """Format a character so labels stay readable for whitespace too."""
    if char == "\n":
        return "\\n"
    if char == "\t":
        return "\\t"
    if char == " ":
        return "space"
    return char


def context_label(text, index):
    previous = "^" if index == 0 else display_char(text[index - 1])
    current = display_char(text[index])
    return f"{previous}{current}@{index}"


def previous_two_label(text, index):
    return "".join(display_char(char) for char in text[index - 2:index])


def timestep_context_label(text, index):
    """Two preceding characters plus the current input character."""
    if index < 0:
        return ""
    if index == 0:
        return f"^{display_char(text[0])}"
    if index == 1:
        return f"{display_char(text[0])}{display_char(text[1])}"
    return previous_two_label(text, index) + display_char(text[index])


def plot_hidden_states_clustermap(text, hidden_states, save_path):
    """Heatmap (timesteps × hidden units) with seaborn clustermap layout."""
    n_rows, n_cols = hidden_states.shape
    if n_rows == 0:
        return

    row_labels = [timestep_context_label(text, t) for t in range(n_rows)]
    col_labels = [f"h{i}" for i in range(n_cols)]
    data = pd.DataFrame(hidden_states, index=row_labels, columns=col_labels)

    grid = sns.clustermap(
        data,
        method="average",
        metric="euclidean",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        figsize=(max(9, n_cols * 0.55), max(6, n_rows * 0.25)),
        dendrogram_ratio=(0.12, 0.1),
        cbar=False,
        xticklabels=True,
        yticklabels=True,
    )
    grid.ax_heatmap.set_xlabel("hidden unit")
    grid.ax_heatmap.set_ylabel("timestep (prev2 + current)")
    grid.ax_heatmap.tick_params(axis="y", labelsize=7)
    grid.ax_heatmap.tick_params(axis="x", labelsize=8)
    grid.fig.suptitle("Hidden states clustered (timesteps × units)", y=1.02, fontsize=11)
    grid.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(grid.fig)
    print(f"wrote {save_path}")


def pca_2d(points):
    """Project points to two dimensions with PCA using NumPy's SVD."""
    centered = points - np.mean(points, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2].T
    return centered @ components


def mds_2d(points):
    """Classical MDS on euclidean distances (same metric as clustermap)."""
    n = points.shape[0]
    if n < 2:
        return np.zeros((n, 2))

    distances = squareform(pdist(points, metric="euclidean"))
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (distances ** 2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order[:2]], 0.0)
    basis = eigenvectors[:, order[:2]]
    return basis * np.sqrt(eigenvalues)


def plot_learning_curve(model, save_path):
    """Plot per-window training loss recorded during training."""
    if "loss_iterations" not in model:
        print(f"skip {save_path}: re-run min-char-rnn.py to record loss history")
        return

    iters = model["loss_iterations"]
    window = model["loss_window"]

    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.plot(iters, window, color="steelblue", linewidth=1.0)
    ax.set_xlabel("iteration")
    ax.set_ylabel("cross-entropy (sum over BPTT window)")
    ax.set_title("Training loss")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_2d_hidden_state_labels(text, hidden_states, chars, projected, save_path, title, xlabel, ylabel):
    """Scatter points with one prev2+current label per sequence, lines to its points."""
    _ = chars
    if len(text) == 0:
        return

    labels = [timestep_context_label(text, i) for i in range(len(text))]
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab20" if len(unique_labels) <= 20 else "hsv")
    sequence_color = {
        label: cmap(i / max(len(unique_labels) - 1, 1))
        for i, label in enumerate(unique_labels)
    }

    fig, ax = plt.subplots(figsize=(14, 11), constrained_layout=True)

    by_sequence = defaultdict(list)
    for i, label in enumerate(labels):
        by_sequence[label].append(i)

    center = projected.mean(axis=0)
    span = max(
        float(np.ptp(projected[:, 0])),
        float(np.ptp(projected[:, 1])),
        1e-3,
    )
    label_offset = span * 0.14
    text_positions = []

    for label, indices in by_sequence.items():
        color = sequence_color[label]
        points = projected[indices]
        ax.scatter(
            points[:, 0], points[:, 1],
            s=36, color=color, edgecolors="black", linewidths=0.35,
            alpha=0.8, zorder=3,
        )

        centroid = points.mean(axis=0)
        outward = centroid - center
        norm = float(np.linalg.norm(outward))
        if norm < 1e-9:
            outward = np.array([0.0, 1.0])
        else:
            outward = outward / norm
        text_pos = centroid + outward * label_offset
        text_positions.append(text_pos)

        for point in points:
            ax.plot(
                [text_pos[0], point[0]], [text_pos[1], point[1]],
                color=color, alpha=0.5, linewidth=0.9, zorder=1,
            )
        ax.text(
            text_pos[0], text_pos[1], label,
            fontsize=10, color=color, ha="center", va="center",
            bbox=dict(
                boxstyle="round,pad=0.25", facecolor="white",
                edgecolor=color, alpha=0.9, linewidth=0.8,
            ),
            zorder=4,
        )

    all_x = np.concatenate([projected[:, 0], [p[0] for p in text_positions]])
    all_y = np.concatenate([projected[:, 1], [p[1] for p in text_positions]])
    x_pad = max((all_x.max() - all_x.min()) * 0.1, 1e-3)
    y_pad = max((all_y.max() - all_y.min()) * 0.1, 1e-3)
    ax.set_xlim(all_x.min() - x_pad, all_x.max() + x_pad)
    ax.set_ylim(all_y.min() - y_pad, all_y.max() + y_pad)

    ax.axhline(0, color="lightgrey", linewidth=0.6, zorder=0)
    ax.axvline(0, color="lightgrey", linewidth=0.6, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.35)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_per_char_hidden_state_heatmaps(text, hidden_states, chars, save_path, cluster_rows=True):
    """Combined per-input-char heatmaps, rows = hidden units, columns = occurrences."""
    hidden_size = hidden_states.shape[1]
    groups = []

    for char in chars:
        indices = np.array([i for i, text_char in enumerate(text) if i > 0 and text_char == char])
        if len(indices) == 0:
            continue

        rows = hidden_states[indices]
        labels = [context_label(text, int(i)) for i in indices]

        if cluster_rows and len(indices) > 2:
            order = average_linkage_cluster_order(rows)
            rows = rows[order]
            labels = [labels[i] for i in order]
            title_suffix = "clustered by hidden-state similarity"
        else:
            title_suffix = "in sequence order"

        groups.append((char, rows, labels, title_suffix))

    if not groups:
        return

    fig, axes = plt.subplots(
        len(groups), 1,
        figsize=(max(12, max(len(labels) for _, _, labels, _ in groups) * 0.28),
                 max(3, len(groups) * max(2.1, hidden_size * 0.24))),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    last_image = None

    for ax, (char, rows, labels, title_suffix) in zip(axes, groups):
        im = ax.imshow(
            rows.T,
            aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
            interpolation="nearest", origin="lower",
        )
        last_image = im

        ax.set_yticks(range(hidden_size))
        ax.set_yticklabels([f"h{i}" for i in range(hidden_size)])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=6, rotation=90)
        ax.set_ylabel("hidden unit")
        ax.set_title(
            f"Hidden states for input {display_char(char)!r} "
            f"({len(labels)} occurrences, {title_suffix})"
        )

    axes[-1].set_xlabel("previous + current character @ timestep")
    fig.suptitle("Hidden-state representations grouped by input character", y=0.995)
    fig.colorbar(last_image, ax=axes, fraction=0.015, pad=0.01, label="activation (tanh)")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_pca_context_labels(text, hidden_states, chars, save_path):
    """2D PCA of hidden states; labels show prev2 + current char."""
    if len(text) < 1:
        return
    plot_2d_hidden_state_labels(
        text, hidden_states, chars,
        pca_2d(hidden_states),
        save_path,
        title="2D PCA (color + label = prev2+current trigram)",
        xlabel="PC1",
        ylabel="PC2",
    )


def plot_mds_context_labels(text, hidden_states, chars, save_path):
    """2D MDS from euclidean distances (same metric as hierarchical clustermap)."""
    if len(text) < 2:
        return
    plot_2d_hidden_state_labels(
        text, hidden_states, chars,
        mds_2d(hidden_states),
        save_path,
        title="2D MDS (color + label = prev2+current trigram)",
        xlabel="MDS 1",
        ylabel="MDS 2",
    )


def char_axis_labels(chars):
    """Tick labels for the vocabulary axis (readable for whitespace)."""
    return [display_char(c) for c in chars]


def symmetric_abs_vmax(*matrices):
    return float(max(np.max(np.abs(m)) for m in matrices))


def plot_learned_weights(model, out_dir):
    """Side-by-side Wxh and Whh heatmaps; hidden units in index order h0..h{n-1}."""
    W_in = np.asarray(model["weights_input_to_hidden"])
    W_rec = np.asarray(model["weights_hidden_to_hidden"])
    chars = model["chars"]
    hidden_size, vocab_size = W_in.shape
    vmax = symmetric_abs_vmax(W_in, W_rec)
    unit_labels = [f"h{i}" for i in range(hidden_size)]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(max(8, vocab_size * 0.5 + hidden_size * 0.5), max(3.5, hidden_size * 0.55)),
        constrained_layout=True,
    )

    axes[0].imshow(
        W_in, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        interpolation="nearest", origin="lower",
    )
    axes[0].set_xticks(range(vocab_size))
    axes[0].set_xticklabels(char_axis_labels(chars), fontsize=8)
    axes[0].set_yticks(range(hidden_size))
    axes[0].set_yticklabels(unit_labels)
    axes[0].set_xlabel("input character")
    axes[0].set_ylabel("hidden unit")
    axes[0].set_title("Input → hidden (Wxh)")

    im1 = axes[1].imshow(
        W_rec, aspect="equal", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        interpolation="nearest", origin="lower",
    )
    axes[1].set_xticks(range(hidden_size))
    axes[1].set_xticklabels(unit_labels, fontsize=7, rotation=90)
    axes[1].set_yticks(range(hidden_size))
    axes[1].set_yticklabels(unit_labels)
    axes[1].set_xlabel("source h (t−1)")
    axes[1].set_ylabel("target h (t)")
    axes[1].set_title("Hidden → hidden (Whh)")

    fig.colorbar(im1, ax=axes, fraction=0.03, pad=0.02, label="weight")
    fig.suptitle("Learned weights (final model)", y=1.02)
    save_path = os.path.join(out_dir, "weights.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_output_probs(text, output_probs, chars, save_path):
    """Heatmap of P(next char) over time; overlay the true next char."""
    vocab_size = len(chars)
    length = len(text)
    targets = list(text[1:]) + [text[0]]
    target_indices = np.array([chars.index(c) for c in targets])

    fig, ax = plt.subplots(figsize=(max(12, length * 0.15), 4))
    im = ax.imshow(
        output_probs.T,
        aspect="auto", cmap="viridis", vmin=0, vmax=1,
        interpolation="nearest", origin="lower",
    )

    ax.set_yticks(range(vocab_size))
    ax.set_yticklabels(chars)
    ax.set_xticks(range(length))
    ax.set_xticklabels(list(text), fontsize=7)
    ax.set_xlabel("timestep / input character")
    ax.set_ylabel("predicted next char")
    ax.set_title("P(next char | input so far)  --  red dots = actual next char")

    ax.scatter(
        np.arange(length), target_indices,
        color="red", s=18, edgecolor="white", linewidth=0.5, zorder=3,
    )

    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="probability")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"wrote {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="model.npz")
    parser.add_argument("--input", default="input.txt")
    parser.add_argument("--length", type=int, default=50,
                        help="how many characters of the corpus to visualize (default: 50)")
    parser.add_argument("--out-dir", default="plots")
    parser.add_argument("--no-cluster-per-char", action="store_true",
                        help="keep per-character heatmap rows in sequence order")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model = load_model(args.model)
    print(f"loaded model: hidden_size={model['hidden_size']}, "
          f"vocab_size={model['vocab_size']}, chars={''.join(model['chars'])}")

    plot_learned_weights(model, args.out_dir)
    plot_learning_curve(
        model,
        save_path=os.path.join(args.out_dir, "learning_curve.png"),
    )

    with open(args.input, "r") as f:
        text = f.read()[: args.length]
    print(f"running forward pass over {len(text)} characters of {args.input}")

    hidden_states, output_probs = forward_pass(model, text)
    targets = list(text[1:]) + [text[0]]

    plot_hidden_states_heatmap(
        text, hidden_states,
        save_path=os.path.join(args.out_dir, "hidden_states_heatmap.png"),
    )

    plot_output_probs(
        text, output_probs, model["chars"],
        save_path=os.path.join(args.out_dir, "output_probabilities.png"),
    )

    plot_per_char_hidden_state_heatmaps(
        text, hidden_states, model["chars"],
        save_path=os.path.join(args.out_dir, "hidden_states_by_input_char.png"),
        cluster_rows=not args.no_cluster_per_char,
    )

    plot_pca_context_labels(
        text, hidden_states, model["chars"],
        save_path=os.path.join(args.out_dir, "hidden_states_pca_context_labels.png"),
    )

    plot_mds_context_labels(
        text, hidden_states, model["chars"],
        save_path=os.path.join(args.out_dir, "hidden_states_mds_context_labels.png"),
    )

    plot_hidden_states_clustermap(
        text, hidden_states,
        save_path=os.path.join(args.out_dir, "hidden_states_clustermap.png"),
    )

    if model["hidden_size"] == 2:
        plot_state_trajectory(
            hidden_states,
            color_by_chars=list(text),
            chars=model["chars"],
            title=f"Hidden state trajectory over {len(text)} chars (colored by INPUT char)",
            save_path=os.path.join(args.out_dir, "hidden_states_trajectory.png"),
        )
        plot_state_trajectory(
            hidden_states,
            color_by_chars=targets,
            chars=model["chars"],
            title=f"Hidden state trajectory over {len(text)} chars (colored by TARGET / next char)",
            save_path=os.path.join(args.out_dir, "hidden_states_by_target.png"),
        )

    correct = np.sum(np.argmax(output_probs, axis=1) ==
                     np.array([model["chars"].index(c) for c in targets]))
    print(f"top-1 next-char accuracy over the {len(text)}-char window: "
          f"{correct}/{len(text)} = {100*correct/len(text):.1f}%")


if __name__ == "__main__":
    main()
