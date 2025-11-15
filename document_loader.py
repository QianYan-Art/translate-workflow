import os
from typing import Optional

def load_document(path: str, pdf_recognition: Optional[dict] = None) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    if ext == ".pdf":
        mode = None
        if pdf_recognition and isinstance(pdf_recognition, dict):
            mode = str(pdf_recognition.get("mode") or "").lower() or "auto"
        if mode and mode != "none":
            try:
                from pdf_recognizer import recognize_pdf
                return recognize_pdf(path, pdf_recognition or {})
            except Exception:
                pass
        try:
            from pdfminer.high_level import extract_text
            text = extract_text(path) or ""
            return text.replace("\r\n", "\n").replace("\r", "\n")
        except Exception:
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(path)
                parts = []
                for page in reader.pages:
                    parts.append(page.extract_text() or "")
                return "\n\n".join(parts)
            except Exception as e2:
                raise RuntimeError(f"Failed to load PDF: {e2}")
    if ext == ".docx":
        try:
            from docx import Document
        except Exception as e:
            raise RuntimeError("python-docx is required to read .docx files")
        doc = Document(path)
        paras = []
        for p in getattr(doc, "paragraphs", []):
            paras.append(p.text or "")
        return "\n".join(paras)
    raise ValueError(f"Unsupported input format: {ext}. Please use .txt, .md, .pdf or .docx")