"""Avaliação de fitness para o problema de predição de sequências.

A rede recebe uma janela de WINDOW_SIZE valores normalizados e deve prever
o próximo valor. O fitness é 1 / (1 + MSE), variando de 0 (péssimo) a 1 (perfeito).

IMPORTANTE: WINDOW_SIZE deve coincidir com num_inputs no arquivo neat.cfg.
"""

from __future__ import annotations

import numpy as np
import neat

from datasets.sequence_builder import SequenceSample

# Tamanho da janela de entrada — deve ser igual a num_inputs em neat.cfg
WINDOW_SIZE: int = 10


def _normalize(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Z-score normaliza a sequência. Retorna (y_norm, mean, std)."""
    mean = float(np.mean(y))
    std = float(np.std(y)) + 1e-8
    return (y - mean) / std, mean, std


def evaluate_genome(
    genome: neat.DefaultGenome,
    config: neat.Config,
    sequences: list[SequenceSample],
) -> float:
    """Avalia um único genoma calculando o MSE médio sobre todas as sequências.

    Para cada sequência:
      1. Normaliza os valores (z-score)
      2. Desliza uma janela de tamanho WINDOW_SIZE ao longo dos valores
      3. Pede à rede que preveja o próximo valor
      4. Acumula o erro quadrático

    Retorna fitness = 1 / (1 + MSE_médio).
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    total_squared_error = 0.0
    n_predictions = 0

    for sample in sequences:
        y = sample.y_values

        if len(y) <= WINDOW_SIZE:
            continue

        if not np.isfinite(y).all():
            continue

        y_norm, _, _ = _normalize(y)

        for i in range(len(y_norm) - WINDOW_SIZE):
            window = y_norm[i : i + WINDOW_SIZE].tolist()
            expected = float(y_norm[i + WINDOW_SIZE])
            predicted = net.activate(window)[0]
            total_squared_error += (predicted - expected) ** 2
            n_predictions += 1

    if n_predictions == 0:
        return 0.0

    mse = total_squared_error / n_predictions
    return 1.0 / (1.0 + mse)


def make_eval_function(sequences: list[SequenceSample]):
    """Retorna uma função compatível com neat.Population.run().

    O neat-python chama eval_genomes(genomes, config) a cada geração,
    onde genomes é uma lista de (genome_id, genome).
    """

    def eval_genomes(
        genomes: list[tuple[int, neat.DefaultGenome]],
        config: neat.Config,
    ) -> None:
        for _genome_id, genome in genomes:
            genome.fitness = evaluate_genome(genome, config, sequences)

    return eval_genomes


def predict_sequence(
    net: neat.nn.FeedForwardNetwork,
    y_known: np.ndarray,
    n_steps: int = 1,
) -> np.ndarray:
    """Usa a rede para prever n_steps valores além de y_known.

    Requer len(y_known) >= WINDOW_SIZE. A predição é feita no espaço
    normalizado e depois desnormalizada para a escala original.
    """
    if len(y_known) < WINDOW_SIZE:
        raise ValueError(f"y_known precisa ter pelo menos {WINDOW_SIZE} valores")

    y_norm, mean, std = _normalize(y_known)
    window = list(y_norm[-WINDOW_SIZE:])
    predictions = []

    for _ in range(n_steps):
        pred_norm = net.activate(window)[0]
        pred = pred_norm * std + mean
        predictions.append(pred)
        window = window[1:] + [pred_norm]

    return np.array(predictions)
