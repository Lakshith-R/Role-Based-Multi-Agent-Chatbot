from PyPDF2 import PdfReader # pylint: disable=import-error

def parse_uploaded_pdf(uploaded_file) -> str:
    """
    Extracts all text from an uploaded curriculum PDF file.

    Args:
        uploaded_file (UploadedFile): Streamlit or other file-like object.

    Returns:
        str: Combined text from all pages.
    """
    pdf = PdfReader(uploaded_file)
    text = "\n".join([
        page.extract_text()
        for page in pdf.pages
        if page.extract_text()
    ])
    return text
