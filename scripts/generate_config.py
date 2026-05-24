"""Gera um arquivo de configuracao minimo do NEAT a partir de um exemplo.

Uso:
    python scripts/generate_config.py
    python scripts/generate_config.py --input config/neat_example.cfg
    python scripts/generate_config.py --output config/neat.cfg
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _strip_config_comments(text: str) -> list[str]:
    """Remove comentarios e linhas vazias, mantendo secoes e pares chave=valor."""
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
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
        line = line.strip()
        if not line:
            continue

        cleaned_lines.append(line)

    return cleaned_lines


def _read_example(example_path: Path) -> str:
    try:
        return example_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return example_path.read_text(encoding="latin-1")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera um arquivo de configuracao minimo do NEAT a partir de um exemplo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        "-i",
        default="config/neat_example.cfg",
        help="Caminho do arquivo de exemplo (padrao: config/neat_example.cfg)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="config/neat.cfg",
        help="Caminho do arquivo de saida (padrao: config/neat.cfg)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Sobrescreve o arquivo mesmo se já existir",
    )
    args = parser.parse_args()

    example_path = Path(args.input)
    output = Path(args.output)

    if not example_path.exists():
        raise SystemExit(f"Arquivo de exemplo nao encontrado: {example_path}")

    if output.exists() and not args.force:
        print(f"Arquivo já existe: {output}")
        print("Use --force para sobrescrever.")
        return

    output.parent.mkdir(parents=True, exist_ok=True)

    raw_example = _read_example(example_path)
    cleaned_lines = _strip_config_comments(raw_example)
    if not cleaned_lines:
        raise SystemExit(f"Arquivo de exemplo vazio ou sem chaves: {example_path}")

    output.write_text(
        "\n".join(cleaned_lines) + "\n", encoding="cp1252", errors="replace"
    )

    print(f"Config gerada em: {output}")
    print()
    print("Proximos passos:")
    print("  1. Edite config/neat_example.cfg e gere novamente")
    print("  2. Ajuste num_inputs para coincidir com WINDOW_SIZE em evaluator.py")
    print("  3. Ajuste pop_size e fitness_threshold conforme seus experimentos")


if __name__ == "__main__":
    main()
