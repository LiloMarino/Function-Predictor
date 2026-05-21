"""Utilitários para configurar e rodar o treino do NEAT."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Callable

import neat


def _strip_config_comments(text: str) -> str:
    """Remove comentarios de um arquivo INI sem tocar nos valores."""
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.lstrip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if stripped.startswith("#") or stripped.startswith(";"):
            continue

        cut_at: int | None = None
        for idx, ch in enumerate(raw_line):
            if ch in "#;" and (idx == 0 or raw_line[idx - 1].isspace()):
                cut_at = idx
                break

        line = raw_line
        if cut_at is not None:
            line = raw_line[:cut_at].rstrip()
            if not line:
                continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines) + "\n"


def _ensure_ansi_config(source_path: Path) -> Path:
    """Gera uma copia ANSI (cp1252) sem comentarios para evitar decode no Windows."""
    ansi_path = source_path.with_name(source_path.name + ".ansi")
    try:
        if (
            ansi_path.exists()
            and ansi_path.stat().st_mtime >= source_path.stat().st_mtime
        ):
            return ansi_path
    except OSError:
        pass

    try:
        text = source_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = source_path.read_text(encoding="latin-1")

    cleaned = _strip_config_comments(text)
    ansi_path.write_text(cleaned, encoding="cp1252", errors="replace")
    return ansi_path


def load_config(config_path: str | Path = "config/neat.cfg") -> neat.Config:
    """Carrega o arquivo de configuracao do NEAT."""
    config_path = Path(config_path)
    try:
        return neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(config_path),
        )
    except UnicodeDecodeError:
        ansi_path = _ensure_ansi_config(config_path)
        return neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(ansi_path),
        )


def create_population(config: neat.Config) -> neat.Population:
    """Cria uma nova população a partir da config."""
    return neat.Population(config)


def restore_checkpoint(
    checkpoint_path: str | Path,
    config: neat.Config,
) -> neat.Population:
    """Restaura uma população de um checkpoint salvo pelo Checkpointer."""
    return neat.Checkpointer.restore_checkpoint(str(checkpoint_path))


def add_reporters(
    population: neat.Population,
    checkpoint_dir: str | Path | None = "checkpoints",
    checkpoint_interval: int = 10,
) -> neat.StatisticsReporter:
    """Adiciona reporters padrão à população e retorna o StatisticsReporter.

    Reporters adicionados:
      - StdOutReporter: imprime progresso a cada geração no console
      - StatisticsReporter: coleta métricas para plotar depois
      - Checkpointer: salva snapshots periódicos (se checkpoint_dir fornecido)
    """
    population.add_reporter(neat.StdOutReporter(True))

    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        prefix = str(Path(checkpoint_dir) / "neat-checkpoint-")
        population.add_reporter(
            neat.Checkpointer(
                generation_interval=checkpoint_interval,
                filename_prefix=prefix,
            )
        )

    return stats


def train(
    eval_function: Callable,
    config_path: str | Path = "config/neat.cfg",
    n_generations: int = 100,
    checkpoint_dir: str | Path | None = "checkpoints",
    checkpoint_interval: int = 10,
) -> tuple[neat.DefaultGenome, neat.StatisticsReporter, neat.Config]:
    """Executa o ciclo completo de treino do NEAT.

    Retorna (melhor_genoma, estatísticas, config).

    Exemplo de uso no notebook:
        eval_fn = make_eval_function(sequences)
        winner, stats, config = train(eval_fn, n_generations=50)
    """
    config = load_config(config_path)
    population = create_population(config)
    stats = add_reporters(population, checkpoint_dir, checkpoint_interval)
    winner = population.run(eval_function, n_generations)
    return winner, stats, config


def save_genome(genome: neat.DefaultGenome, path: str | Path) -> None:
    """Serializa um genoma para disco com pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(genome, f)


def load_genome(path: str | Path) -> neat.DefaultGenome:
    """Carrega um genoma serializado pelo save_genome."""
    with open(path, "rb") as f:
        return pickle.load(f)
