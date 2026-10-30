from pathlib import Path
from app.ingestion.extractor import extract_pdf_text


def main():
    raw_path = input("Paste the full path to your PDF, then press Enter: ").strip().strip('"')
    pdf_path = Path(raw_path)

    if not pdf_path.is_file():
        print(f"\nERROR: PDF not found:\n{pdf_path}")
        print("\nTip: paste the complete file path, including the .pdf filename.")
        input("\nPress Enter to exit...")
        return

    try:
        pages = extract_pdf_text(pdf_path)
    except Exception as exc:
        print(f"\nERROR: PDF extraction failed: {exc}")
        input("\nPress Enter to exit...")
        return

    print(f"\nPages with extractable text: {len(pages)}")
    keywords = (
        "project guide",
        "guide name",
        "guided by",
        "under the guidance",
        "guide:",
        "supervisor",
        "project supervisor",
        "guide name:",
    )

    matches = []
    for page in pages:
        if any(keyword in page.text.lower() for keyword in keywords):
            matches.append(page)

    if not matches:
        print("\nNo guide-related keywords were found in extracted PDF text.")
        print("The name may be inside an image/scanned page, or the PDF wording may differ.")
    else:
        for page in matches:
            print("\n" + "=" * 24 + f" PDF PAGE {page.page_number} " + "=" * 24)
            print(page.text)

    print("\nDone. Copy the complete output from this terminal and send it here.")
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
