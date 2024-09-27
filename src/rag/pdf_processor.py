import os
from PyPDF2 import PdfReader
from tests.text_processor import process_text, init_database, load_config
from src.common.logger_config import LoggerConfig

config = load_config()

# Configure logging
logger = LoggerConfig.setup_logger(__name__)

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def process_pdf_files(folder_path):
    init_database()

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            content = extract_text_from_pdf(file_path)
            
            process_text(content, filename)

    logger.info("PDF processing, chunking, and embedding generation complete.")

if __name__ == "__main__":
    pdf_folder = config['pdf_folder']
    process_pdf_files(pdf_folder)