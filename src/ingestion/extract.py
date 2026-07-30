from pathlib import Path

import pymupdf4llm



def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    pdf_candidates = [
        project_root / "data" / "raw" / "top_11_multi_vendor_marketplace_platforms_for_ecommerce.pdf",
        
    ]
    pdf_path = next((path for path in pdf_candidates if path.exists()), None)

    if pdf_path is None:
        raise FileNotFoundError(
            "Could not find the PDF file. Checked: "
            + ", ".join(str(path) for path in pdf_candidates)
        )

    output_path = project_root / "data" / "processed" / "extracted_output.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(" Extracting text and converting to Markdown format...")
    markdown_text = pymupdf4llm.to_markdown(str(pdf_path))

    with output_path.open("w", encoding="utf-8") as f:
        f.write(markdown_text)

    print(f" Extraction complete! Saved to '{output_path}'.")



if __name__ == "__main__":
    main()