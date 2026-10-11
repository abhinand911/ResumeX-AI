import io
from pypdf import PdfReader
from docx import Document


def extract_text_from_pdf(file):
    """Extract text from an uploaded PDF file."""
    try:
        pdf_bytes = file.read()
        pdf_file = io.BytesIO(pdf_bytes)

        reader = PdfReader(pdf_file)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text.strip()

    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")


def extract_text_from_docx(file):
    """Extract text from an uploaded DOCX file."""
    try:
        docx_bytes = file.read()
        docx_file = io.BytesIO(docx_bytes)

        document = Document(docx_file)

        text = ""

        for paragraph in document.paragraphs:
            if paragraph.text.strip():
                text += paragraph.text + "\n"

        return text.strip()

    except Exception as e:
        raise Exception(f"Error reading DOCX: {str(e)}")


def extract_resume_text(file):
    """
    Detect the uploaded file type and extract its text.
    Supports PDF and DOCX.
    """

    file_name = file.name.lower()

    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(file)

    elif file_name.endswith(".docx"):
        return extract_text_from_docx(file)

    else:
        raise ValueError(
            "Unsupported file format. Please upload a PDF or DOCX resume."
        )