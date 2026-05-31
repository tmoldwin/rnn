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

  5) hidden_states_pca_context_labels.png
       2D PCA projection of hidden states. Each state is shown as a text
       annotation containing the two preceding characters, colored by the
       current input character.

Usage:
    python visualize.py                  # default: 200 chars
    python visualize.py --length 60
    python visualize.py --length 200 --out-dir plots
"""

from __future__ import annotations

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np


def load_model(path: str = "model.npz"):
    data = np.load(path, allow_pickle=False)
    return {
        "weights_input_to_hidden":  data["weights_input_to_hidden"],
        "weights_hidden_to_hidden": data["weights_hidden_to_hidden"],
        "weights_hidden_to_output": data["weights_hidden_to_output"],
        "bias_hidden":              data["bias_hidden"],
        "bias_output":              data["bias_output"],
        "chars":                    [str(c) for c in data["chars"]],
        "hidden_size":              int(data["hidden_size"]),
        "vocab_size":               int(data["vocab_size"]),
    }


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


def average_linkage_cluster_order(rows):
    """Return row indices ordered by a small average-linkage clustering pass."""
    n_rows = rows.shape[0]
    if n_rows <= 2:
        return list(range(n_rows))

    distances = np.linalg.norm(rows[:, None, :] - rows[None, :, :], axis=2)
    clusters = [{"indices": [i], "members": [i]} for i in range(n_rows)]

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
        merged = {
            "indices": clusters[left]["indices"] + clusters[right]["indices"],
            "members": clusters[left]["members"] + clusters[right]["members"],
        }

        clusters[left] = merged
        del clusters[right]

    return clusters[0]["indices"]


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


def pca_2d(points):
    """Project points to two dimensions with PCA using NumPy's SVD."""
    centered = points - np.mean(points, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2].T
    return centered @ components


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
    """2D PCA of hidden states annotated by the two characters before each state."""
    if len(text) < 3:
        return

    indices = np.arange(2, len(text))
    projected = pca_2d(hidden_states[indices])
    current_chars = [text[int(i)] for i in indices]
    labels = [previous_two_label(text, int(i)) for i in indices]
    cmap = plt.get_cmap("tab10")
    char_to_color = {c: cmap(i % 10) for i, c in enumerate(chars)}

    fig, ax = plt.subplots(figsize=(14, 11), constrained_layout=True)
    ax.scatter(projected[:, 0], projected[:, 1], s=1, alpha=0)

    for (x, y), label, char in zip(projected, labels, current_chars):
        ax.annotate(
            label,
            (x, y),
            color=char_to_color[char],
            fontsize=11,
            ha="center",
            va="center",
            alpha=0.85,
        )

    x_min, x_max = np.min(projected[:, 0]), np.max(projected[:, 0])
    y_min, y_max = np.min(projected[:, 1]), np.max(projected[:, 1])
    x_pad = max((x_max - x_min) * 0.08, 1e-3)
    y_pad = max((y_max - y_min) * 0.08, 1e-3)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=char_to_color[c], label=repr(c))
        for c in chars
        if c in current_chars
    ]
    ax.legend(handles=handles, title="current char", loc="best", framealpha=0.9)
    ax.axhline(0, color="lightgrey", linewidth=0.6, zorder=0)
    ax.axvline(0, color="lightgrey", linewidth=0.6, zorder=0)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(
        "2D PCA of hidden states "
        "(annotation = previous two chars, color = current char)"
    )
    ax.grid(True, linestyle=":", alpha=0.35)
    fig.savefig(save_path, dpi=300)
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
    parser.add_argument("--length", type=int, default=200,
                        help="how many characters of the corpus to visualize (default: 200)")
    parser.add_argument("--out-dir", default="plots")
    parser.add_argument("--no-cluster-per-char", action="store_true",
                        help="keep per-character heatmap rows in sequence order")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model = load_model(args.model)
    print(f"loaded model: hidden_size={model['hidden_size']}, "
          f"vocab_size={model['vocab_size']}, chars={''.join(model['chars'])}")

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
