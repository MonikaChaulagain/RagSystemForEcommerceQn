import json
import os
from pathlib import Path
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


def load_markdown(file_path: str) -> str:
    """Load extracted markdown text from file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Markdown file not found at '{file_path}'. "
            "Please run 'src/ingestion/extract.py' first."
        )
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def add_header_context(chunks):
    """
    Prepend the header hierarchy to each chunk's content so the embedding
    model actually sees the topic, not just an isolated fragment.
    Also stores a clean 'section_path' string in metadata for filtering/display.
    """
    enriched = []
    for chunk in chunks:
        headers = [chunk.metadata.get(f"Header {i}") for i in (1, 2, 3)]
        headers = [h for h in headers if h]
        # de-dupe while preserving order (Header 1 sometimes repeats as Header 2)
        section_path = " > ".join(dict.fromkeys(headers))

        if section_path:
            chunk.page_content = f"{section_path}\n{chunk.page_content}"

        chunk.metadata["section_path"] = section_path
        enriched.append(chunk)
    return enriched


def chunk_markdown(
    markdown_content: str,
    chunk_size: int = 500,       # smaller, more focused chunks
    chunk_overlap: int = 100,
):
    """
    Two-stage chunking strategy for RAG:
    1. Structure-aware splitting based on Markdown Headings (#, ##, ###).
    2. Size-based recursive splitting to ensure chunks fit embedding context windows.
    3. Header-context injection so embeddings retain topic information.
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    header_splits = markdown_splitter.split_text(markdown_content)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    final_chunks = text_splitter.split_documents(header_splits)
    final_chunks = add_header_context(final_chunks)
    return final_chunks


def save_chunks(chunks, output_path: str):
    """Save chunked documents to disk as JSON for inspection & embedding."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    chunk_data = []
    for i, chunk in enumerate(chunks):
        chunk_data.append(
            {
                "chunk_id": i,
                "content": chunk.page_content,
                "metadata": chunk.metadata,
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved {len(chunks)} chunks to '{output_path}'.")


def main():
    input_md_path = os.path.join("data", "processed", "extracted_output.md")
    output_json_path = os.path.join("data", "processed", "chunks.json")

    print(" Loading extracted Markdown text...")
    markdown_text = load_markdown(input_md_path)

    print(" Applying structure-aware chunking...")
    chunks = chunk_markdown(markdown_text)

    print(f" Total Chunks Generated: {len(chunks)}")

    if chunks:
        print("\n--- Chunk #0 Preview ---")
        print(f"Metadata: {chunks[0].metadata}")
        print(f"Content:\n{chunks[0].page_content[:300]}...\n")

    save_chunks(chunks, output_json_path)


if __name__ == "__main__":
    main()