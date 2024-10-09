import os
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
import pandas as pd
from db_manager import DBManager
from logger_config import LoggerConfig
import json
from graph_rag_project.pdf_extractor import PDFExtractor

config = LoggerConfig.load_config()
logger = LoggerConfig.setup_logger(__name__)

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def extract_tables_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    tables = []
    for page in doc:
        page_tables = page.find_tables()
        for table in page_tables.tables:
            extracted_table = table.extract()
            if extracted_table:
                headers = extracted_table[0]
                data = extracted_table[1:]
                df = pd.DataFrame(data, columns=headers)
                tables.append({
                    'page': page.number + 1,
                    'headers': headers,
                    'content': df.to_dict('records')
                })
    return tables

def generate_html(text, tables):
    html = "<html><body>"
    html += f"<h1>Extracted Text</h1><p>{text}</p>"
    html += "<h1>Extracted Tables</h1>"
    for i, table in enumerate(tables):
        html += f"<h2>Table {i+1} (Page {table['page']})</h2>"
        df = pd.DataFrame(table['content'])
        html += df.to_html(index=False)
    html += "</body></html>"
    return html

def save_sales_report_to_db(tables, db_manager):
    for table in tables:
        if any('sales' in header.lower() for header in table['headers']):
            db_manager.create_sales_report_table()
            db_manager.insert_sales_report_data(table['content'])
            logger.info(f"Sales Report data saved to database. Rows inserted: {len(table['content'])}")
            break
    else:
        logger.warning("No Sales Report found in extracted content")

def process_pdf_files(folder_path):
    db_manager = DBManager('sales_report.db')

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            
            # Extract text and tables
            text = extract_text_from_pdf(file_path)
            tables = extract_tables_from_pdf(file_path)
            
            # Generate HTML
            html_content = generate_html(text, tables)
            with open("output.html", "w", encoding='utf-8') as f:
                f.write(html_content)
            
            # Save sales report to database
            save_sales_report_to_db(tables, db_manager)
            
            # Print all rows from sales_report table
            all_sales_data = db_manager.get_all_sales_report_data()
            logger.info(f"All entries in the sales_report table (count: {len(all_sales_data)}):")
            for row in all_sales_data:
                logger.info(json.dumps(row, indent=2))

    logger.info("PDF processing, HTML generation, and database operations complete.")

if __name__ == "__main__":
    pdf_folder = config['pdf_folder']
    process_pdf_files(pdf_folder)