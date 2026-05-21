"""Visualizações para análise de treino e resultado do NEAT."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import neat

from datasets.sequence_builder import SequenceSample
from neat_core.evaluator import WINDOW_SIZE


# ---------------------------------------------------------------------------
# Gráficos de treino
# ---------------------------------------------------------------------------


def plot_fitness(
    stats: neat.StatisticsReporter,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plota o fitness máximo e médio ao longo das gerações."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    best = [g.fitness for g in stats.most_fit_genomes]
    mean = stats.get_fitness_mean()
    stdev = stats.get_fitness_stdev()
    gens = range(len(best))

    mean_arr = np.array(mean)
    stdev_arr = np.array(stdev)

    ax.plot(gens, best, "b-", linewidth=2, label="Melhor fitness")
    ax.plot(gens, mean_arr, "r--", linewidth=1.5, label="Fitness médio")
    ax.fill_between(
        gens,
        mean_arr - stdev_arr,
        mean_arr + stdev_arr,
        alpha=0.15,
        color="red",
        label="±1 desvio padrão",
    )
    ax.set_xlabel("Geração")
    ax.set_ylabel("Fitness  [1 / (1 + MSE)]")
    ax.set_title("Fitness ao longo do treino")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    return ax


def plot_species(
    stats: neat.StatisticsReporter,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plota o número de espécies por geração (gráfico de área empilhada)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    species_sizes = stats.get_species_sizes()
    if not species_sizes:
        ax.text(0.5, 0.5, "Sem dados de espécies", ha="center", va="center")
        return ax

    n_gens = len(species_sizes)
    n_species = max(len(s) for s in species_sizes)

    # Preenche com zeros para espécies que surgiram/desapareceram
    matrix = np.zeros((n_gens, n_species))
    for g, sizes in enumerate(species_sizes):
        for s, count in enumerate(sizes):
            matrix[g, s] = count

    gens = np.arange(n_gens)
    colors = plt.cm.tab20(np.linspace(0, 1, n_species))
    ax.stackplot(gens, matrix.T, colors=colors, alpha=0.8)
    ax.set_xlabel("Geração")
    ax.set_ylabel("Indivíduos por espécie")
    ax.set_title("Distribuição de espécies ao longo do treino")
    ax.grid(True, alpha=0.3, axis="y")
    return ax


def plot_training_summary(stats: neat.StatisticsReporter) -> plt.Figure:
    """Figura com fitness e espécies lado a lado."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    plot_fitness(stats, ax=ax1)
    plot_species(stats, ax=ax2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Visualização da topologia da rede
# ---------------------------------------------------------------------------


def draw_network(
    config: neat.Config,
    genome: neat.DefaultGenome,
    ax: plt.Axes | None = None,
    node_radius: float = 0.04,
) -> plt.Axes:
    """Desenha a topologia da rede evoluída.

    - Nós de entrada: azul claro (esquerda)
    - Nós ocultos:    amarelo    (centro)
    - Nós de saída:   verde claro (direita)
    - Conexões positivas: azul
    - Conexões negativas: vermelho
    - Conexões desabilitadas: cinza tracejado
    """
    gc = config.genome_config
    input_keys: list[int] = gc.input_keys
    output_keys: list[int] = gc.output_keys
    hidden_keys: list[int] = [
        k for k in genome.nodes if k not in input_keys and k not in output_keys
    ]

    n_in = len(input_keys)
    n_out = len(output_keys)
    n_hid = len(hidden_keys)

    # Posições: inputs à esq., hidden no meio, outputs à dir.
    positions: dict[int, tuple[float, float]] = {}

    def _ys(n: int) -> list[float]:
        if n == 1:
            return [0.5]
        return [i / (n - 1) for i in range(n)]

    for k, y in zip(input_keys, _ys(n_in)):
        positions[k] = (0.0, y)

    for k, y in zip(output_keys, _ys(n_out)):
        positions[k] = (1.0, y)

    for k, y in zip(hidden_keys, _ys(max(n_hid, 1))):
        positions[k] = (0.5, y)

    # Tamanho da figura baseado no número de inputs
    fig_h = max(5, n_in * 0.7)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, fig_h))

    # Conexões
    max_weight = max(
        (abs(cg.weight) for cg in genome.connections.values()), default=1.0
    )

    for cg in genome.connections.values():
        src, dst = cg.key
        if src not in positions or dst not in positions:
            continue
        x0, y0 = positions[src]
        x1, y1 = positions[dst]
        if cg.enabled:
            color = "#2166ac" if cg.weight > 0 else "#d6604d"
            lw = 0.5 + 2.5 * abs(cg.weight) / (max_weight + 1e-9)
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=lw,
                    alpha=0.7,
                ),
            )
        else:
            ax.plot([x0, x1], [y0, y1], ":", color="gray", lw=0.5, alpha=0.4)

    # Nós
    node_colors = {
        "input": "#aec6e8",
        "hidden": "#ffe599",
        "output": "#b6d7a8",
    }

    for key, (x, y) in positions.items():
        if key in input_keys:
            kind = "input"
            label = f"in{abs(key)}"
        elif key in output_keys:
            kind = "output"
            label = f"out{key}"
        else:
            kind = "hidden"
            label = str(key)

        circle = plt.Circle(
            (x, y),
            node_radius,
            color=node_colors[kind],
            ec="black",
            linewidth=0.8,
            zorder=3,
        )
        ax.add_patch(circle)
        fontsize = max(5, min(8, 60 // max(n_in, 1)))
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, zorder=4)

    # Legenda
    legend_handles = [
        mpatches.Patch(color=node_colors["input"], label="Input"),
        mpatches.Patch(color=node_colors["hidden"], label="Hidden"),
        mpatches.Patch(color=node_colors["output"], label="Output"),
        mlines.Line2D([], [], color="#2166ac", label="Peso positivo"),
        mlines.Line2D([], [], color="#d6604d", label="Peso negativo"),
        mlines.Line2D([], [], color="gray", linestyle=":", label="Desabilitada"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"Topologia — {len(genome.nodes)} nós, "
        f"{sum(1 for c in genome.connections.values() if c.enabled)} conexões ativas"
    )
    return ax


# ---------------------------------------------------------------------------
# Gráfico de predição vs. real
# ---------------------------------------------------------------------------


def _normalize(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(y))
    std = float(np.std(y)) + 1e-8
    return (y - mean) / std, mean, std


def plot_prediction(
    net: neat.nn.FeedForwardNetwork,
    sequence: SequenceSample,
    ax: plt.Axes | None = None,
    window_size: int = WINDOW_SIZE,
) -> plt.Axes:
    """Plota valores reais vs. preditos pela rede em uma sequência."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    y = sequence.y_values
    y_norm, mean, std = _normalize(y)

    actual, predicted = [], []
    for i in range(len(y_norm) - window_size):
        window = y_norm[i : i + window_size].tolist()
        pred_norm = net.activate(window)[0]
        actual.append(float(y[i + window_size]))
        predicted.append(pred_norm * std + mean)

    actual_arr = np.array(actual)
    predicted_arr = np.array(predicted)
    mse = float(np.mean((actual_arr - predicted_arr) ** 2))

    indices = range(window_size, len(y))
    ax.plot(range(len(y)), y, "k-", alpha=0.3, label="Sequência completa")
    ax.plot(indices, actual_arr, "b-", linewidth=1.5, label="Real")
    ax.plot(indices, predicted_arr, "r--", linewidth=1.5, label="Predito")
    ax.set_xlabel("Passo")
    ax.set_ylabel("Valor")
    ax.set_title(f"{sequence.function_name}  |  MSE = {mse:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_extrapolation(
    net: neat.nn.FeedForwardNetwork,
    sequence: SequenceSample,
    n_extra: int = 10,
    ax: plt.Axes | None = None,
    window_size: int = WINDOW_SIZE,
) -> plt.Axes:
    """Plota a sequência conhecida + extrapolação além dos dados de treino.

    Mostra até onde a rede consegue extrapolar sem ver novos valores reais.
    """
    from neat_core.evaluator import predict_sequence

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    y = sequence.y_values
    extra = predict_sequence(net, y, n_steps=n_extra)

    n = len(y)
    ax.plot(range(n), y, "b-", linewidth=1.5, label="Sequência conhecida")
    ax.plot(
        range(n, n + n_extra),
        extra,
        "r--",
        linewidth=1.5,
        marker="o",
        markersize=4,
        label=f"Extrapolação ({n_extra} passos)",
    )
    ax.axvline(x=n - 0.5, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Passo")
    ax.set_ylabel("Valor")
    ax.set_title(f"Extrapolação — {sequence.function_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax
