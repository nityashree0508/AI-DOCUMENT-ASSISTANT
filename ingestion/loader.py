from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


class PDFLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_documents(self):
        documents = []

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Directory '{self.data_dir}' does not exist."
            )

        pdf_files = list(self.data_dir.glob("*.pdf"))

        if len(pdf_files) == 0:
            raise FileNotFoundError(
                "No PDF files found in data folder."
            )

        print(f"Found {len(pdf_files)} PDF(s).\n")

        for pdf in pdf_files:
            print(f"Loading {pdf.name}")

            loader = PyPDFLoader(str(pdf))

            documents.extend(loader.load())

        print(f"\nLoaded {len(documents)} pages.")

        return documents