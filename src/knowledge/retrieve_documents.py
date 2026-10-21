"""Recuperação lexical de evidências para o RAG do Company Smart Loyalty."""

import argparse
import json
import re
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
MAX_CHUNK_CHARS = 1200

STOPWORDS = {
    "a", "as", "o", "os", "um", "uma", "uns", "umas",
    "de", "da", "das", "do", "dos", "e", "em", "no",
    "na", "nos", "nas", "por", "para", "com", "que",
    "se", "ao", "aos", "ou", "como", "mais", "sua",
    "seu", "suas", "seus", "qual", "quais", "sobre",
    "ser", "sao", "são", "foi", "tem", "ter", "pode",
    "podem", "entre", "tambem", "também",
}


def normalize(text: str) -> str:
    """Normaliza acentos e caixa para melhorar a busca em português."""
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in text if not unicodedata.combining(char))


def tokenize(text: str) -> list[str]:
    """Extrai termos relevantes, descartando palavras muito comuns."""
    words = re.findall(r"\b[a-zA-Z0-9]+\b", normalize(text))
    return [
        word for word in words
        if len(word) > 1 and word not in STOPWORDS
    ]



def split_document(text: str, source: str) -> list[dict]:
    """Divide Markdown em trechos, preservando títulos e listas."""
    chunks = []
    section = "Introdução"
    buffer = []

    def save_buffer():
        content = "\n".join(buffer).strip()
        if content:
            chunks.append({
                "source": source,
                "section": section,
                "text": content,
            })
        buffer.clear()

    for line in text.splitlines():
        stripped = line.strip()
        heading = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)

        if heading:
            save_buffer()
            section = heading.group(1).strip()
            buffer.append(stripped)
            continue

        if not stripped:
            # Mantém títulos e listas juntos do conteúdo relacionado.
            if buffer:
                previous = buffer[-1]
                is_heading = bool(
                    re.match(r"^\s{0,3}#{1,6}\s+", previous)
                )
                is_list_item = bool(
                    re.match(r"^\s*[-*+]\s+", previous)
                    or re.match(r"^\s*\d+[.)]\s+", previous)
                )

                if not is_heading and not is_list_item:
                    save_buffer()
            continue

        candidate = "\n".join(buffer + [stripped])

        if len(candidate) > MAX_CHUNK_CHARS and buffer:
            save_buffer()

        buffer.append(stripped)

    save_buffer()
    return chunks


def load_documents(documents_dir: Path = DEFAULT_DOCUMENTS_DIR) -> list[dict]:
    """Carrega arquivos Markdown da pasta documental."""
    if not documents_dir.exists():
        raise FileNotFoundError(
            f"Pasta documental não encontrada: {documents_dir}"
        )

    documents = []
    for path in sorted(documents_dir.rglob("*.md")):
        if path.name.lower() == "readme.md":
            continue

        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = path.relative_to(documents_dir).as_posix()
        documents.extend(split_document(content, relative_path))

    return documents



def search_documents(
    query: str,
    documents: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """Retorna trechos ranqueados, priorizando a relevância da seção."""
    if top_k < 1:
        raise ValueError("top_k precisa ser maior que zero.")

    query_terms = set(tokenize(query))
    if not query_terms:
        return []

    results = []

    for chunk in documents:
        section_terms = set(tokenize(chunk["section"]))
        content_terms = set(tokenize(chunk["text"]))

        matched = sorted(
            query_terms & (section_terms | content_terms)
        )

        if not matched:
            continue

        # A seção recebe peso maior que o texto corrido.
        # Isso favorece, por exemplo, a seção "Benefícios mencionados"
        # em perguntas específicas sobre benefícios.
        section_coverage = (
            len(query_terms & section_terms) / len(query_terms)
        )
        content_coverage = (
            len(query_terms & content_terms) / len(query_terms)
        )

        score = (
            0.65 * section_coverage
            + 0.35 * content_coverage
        )

        results.append({
            **chunk,
            "score": round(score, 4),
            "matched_terms": matched,
        })

    results.sort(
        key=lambda item: (
            -item["score"],
            item["source"],
            item["section"],
        )
    )

    return results[:top_k]

def main():
    parser = argparse.ArgumentParser(
        description="Busca evidências na documentação do RAG."
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Pergunta ou termos que deseja pesquisar.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Quantidade máxima de trechos retornados.",
    )
    parser.add_argument(
        "--documents-dir",
        type=Path,
        default=DEFAULT_DOCUMENTS_DIR,
        help="Pasta contendo documentos Markdown.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Imprime o resultado em JSON.",
    )
    args = parser.parse_args()

    documents = load_documents(args.documents_dir)
    results = search_documents(args.query, documents, args.top_k)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    print(f"Trechos indexados em memória: {len(documents)}")
    print(f"Resultados encontrados: {len(results)}")

    for index, result in enumerate(results, start=1):
        print(f"\n[{index}] {result['source']}")
        print(f"Seção: {result['section']}")
        print(f"Pontuação: {result['score']}")
        print(f"Termos encontrados: {', '.join(result['matched_terms'])}")
        print(result["text"])


if __name__ == "__main__":
    main()
