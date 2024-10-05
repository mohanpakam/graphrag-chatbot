import fitz  # PyMuPDF
import pandas as pd
import json
from tabulate import tabulate
from db_manager import DBManager
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class PDFExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(self.pdf_path)
        self.extracted_data = {
            'text': [],
            'tables': []
        }
        self.db_manager = DBManager('sales_report.db')
        self.current_large_table = None

    def extract_content(self):
        self.extract_text_and_tables()
        self.finalize_large_table()

    def extract_text_and_tables(self):
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text, page_tables = self.process_page(page, page_num)
            
            self.extracted_data['text'].append({
                'page': page_num + 1,
                'content': text
            })
            
            self.process_tables(page_tables, page_num)

    def process_page(self, page, page_num):
        text = page.get_text()
        tables = page.find_tables()
        processed_tables = []
        
        for table in tables.tables:
            extracted_table = table.extract()
            if extracted_table:
                bbox = table.bbox
                processed_tables.append({
                    'page': page_num + 1,
                    'content': extracted_table,
                    'bbox': {
                        'x0': bbox[0] if isinstance(bbox, tuple) else bbox.x0,
                        'y0': bbox[1] if isinstance(bbox, tuple) else bbox.y0,
                        'x1': bbox[2] if isinstance(bbox, tuple) else bbox.x1,
                        'y1': bbox[3] if isinstance(bbox, tuple) else bbox.y1
                    }
                })
        
        processed_tables.sort(key=lambda t: (t['bbox']['y0'], t['bbox']['x0']))
        
        logger.info(f"Extracted {len(processed_tables)} tables from page {page_num + 1}")
        return text, processed_tables

    def process_tables(self, page_tables, page_num):
        for table in page_tables:
            if self.is_continuation_of_large_table(table):
                self.continue_large_table(table)
            elif self.is_start_of_large_table(table):
                self.start_large_table(table)
            else:
                # Ensure all tables have a consistent structure
                processed_table = {
                    'page': table['page'],
                    'content': table['content'],
                    'bbox': table['bbox']
                }
                if len(table['content']) > 0:
                    processed_table['headers'] = table['content'][0]
                    processed_table['content'] = table['content'][1:]
                self.extracted_data['tables'].append(processed_table)

    def is_start_of_large_table(self, table):
        return len(table['content']) > 10 and self.current_large_table is None

    def start_large_table(self, table):
        self.current_large_table = {
            'pages': [table['page']],
            'headers': table['content'][0],
            'content': table['content'][1:],
            'bbox': table['bbox']
        }

    def is_continuation_of_large_table(self, table):
        if not self.current_large_table:
            return False
        return len(table['content'][0]) == len(self.current_large_table['headers'])

    def continue_large_table(self, table):
        self.current_large_table['content'].extend(table['content'])
        self.current_large_table['pages'].append(table['page'])

    def finalize_large_table(self):
        if self.current_large_table:
            self.extracted_data['tables'].append(self.current_large_table)
            self.current_large_table = None

    def to_json(self):
        return json.dumps(self.extracted_data, indent=2, default=str)

    def to_html(self):
        html = "<html><body>"
        for text in self.extracted_data['text']:
            html += f"<h2>Page {text['page']}</h2>"
            html += f"<p>{text['content']}</p>"
        
        for table in self.extracted_data['tables']:
            if 'pages' in table:
                html += f"<h3>Table on Page(s) {', '.join(map(str, table['pages']))}</h3>"
            else:
                html += f"<h3>Table on Page {table['page']}</h3>"
            
            if 'headers' in table and table['headers']:
                df = pd.DataFrame(table['content'], columns=table['headers'])
            else:
                df = pd.DataFrame(table['content'])
            
            html += df.to_html(index=False)
        
        html += "</body></html>"
        return html

    def to_markdown(self):
        md = ""
        for text in self.extracted_data['text']:
            md += f"## Page {text['page']}\n\n"
            md += f"{text['content']}\n\n"
        
        for table in self.extracted_data['tables']:
            if 'pages' in table:
                md += f"### Table on Page(s) {', '.join(map(str, table['pages']))}\n\n"
            else:
                md += f"### Table on Page {table['page']}\n\n"
            
            if 'headers' in table and table['headers']:
                df = pd.DataFrame(table['content'], columns=table['headers'])
            else:
                df = pd.DataFrame(table['content'])
            
            md += tabulate(df, headers='keys', tablefmt='pipe') + "\n\n"
        
        return md

def save_sales_report_to_db(extracted_content: Dict[str, Any], db_manager: DBManager):
    sales_data = []
    for table in extracted_content['tables']:
        if 'headers' in table and any('sales' in header.lower() for header in table['headers']):
            for row in table['content']:
                try:
                    sales_row = {
                        'Month': row[0],
                        'Electronics': row[1] if len(row) > 1 else '',
                        'Clothing': row[2] if len(row) > 2 else '',
                        'Food': row[3] if len(row) > 3 else '',
                        'Books': row[4] if len(row) > 4 else '',
                        'Total': row[-1]
                    }
                    sales_data.append(sales_row)
                except Exception as e:
                    logger.error(f"Error processing row: {row}. Error: {str(e)}")
    
    if sales_data:
        db_manager.create_sales_report_table()
        db_manager.insert_sales_report_data(sales_data)
        logger.info(f"Sales Report data saved to database. Rows inserted: {len(sales_data)}")
    else:
        logger.warning("No Sales Report data found in the extracted content")

# Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    pdf_path = "complex_sample.pdf"  # Replace with your PDF path
    extractor = PDFExtractor(pdf_path)
    extractor.extract_content()
    
    # Log extracted content for debugging
    logger.debug(f"Extracted text: {json.dumps(extractor.extracted_data['text'], indent=2)}")
    logger.debug(f"Extracted tables: {json.dumps(extractor.extracted_data['tables'], indent=2)}")

    # Save extracted content in different formats
    with open("output.json", "w") as f:
        f.write(extractor.to_json())
    
    with open("output.html", "w") as f:
        f.write(extractor.to_html())
    
    with open("output.md", "w") as f:
        f.write(extractor.to_markdown())

    # Save Sales Report to database
    save_sales_report_to_db(extractor.extracted_data, extractor.db_manager)

    # Display all entries in the sales_report table
    all_sales_data = extractor.db_manager.get_all_sales_report_data()
    logger.info(f"All entries in the sales_report table (count: {len(all_sales_data)}):")
    for row in all_sales_data:
        logger.info(json.dumps(row, indent=2))