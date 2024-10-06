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
            'tables': []
        }
        self.db_manager = DBManager('sales_report.db')
        self.current_large_table = None

    def extract_content(self):
        self.extract_tables()
        self.finalize_large_table()

    def extract_tables(self):
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            page_tables = self.process_page(page, page_num)
            self.process_tables(page_tables, page_num)

    def process_page(self, page, page_num):
        tables = page.find_tables()
        processed_tables = []
        
        for table in tables.tables:
            extracted_table = table.extract()
            if extracted_table:
                bbox = table.bbox
                processed_tables.append({
                    'page': page_num + 1,
                    'content': self.clean_table_content(extracted_table),
                    'bbox': {
                        'x0': bbox[0] if isinstance(bbox, tuple) else bbox.x0,
                        'y0': bbox[1] if isinstance(bbox, tuple) else bbox.y0,
                        'x1': bbox[2] if isinstance(bbox, tuple) else bbox.x1,
                        'y1': bbox[3] if isinstance(bbox, tuple) else bbox.y1
                    }
                })
        
        processed_tables.sort(key=lambda t: (t['bbox']['y0'], t['bbox']['x0']))
        
        logger.info(f"Extracted {len(processed_tables)} tables from page {page_num + 1}")
        return processed_tables

    def clean_table_content(self, table_content):
        return [
            [str(cell).strip() if cell is not None else '' for cell in row if cell is not None and str(cell).strip()]
            for row in table_content
            if any(cell is not None and str(cell).strip() for cell in row)
        ]

    def process_tables(self, page_tables, page_num):
        for table in page_tables:
            if self.is_continuation_of_large_table(table):
                self.continue_large_table(table)
            elif self.is_start_of_large_table(table):
                self.start_large_table(table)
            else:
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
            'headers': table['content'][0] if table['content'] else [],
            'content': table['content'][1:] if len(table['content']) > 1 else [],
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

    def to_markdown(self):
        md = ""
        for table in self.extracted_data['tables']:
            if 'pages' in table:
                md += f"### Table on Page(s) {', '.join(map(str, table['pages']))}\n\n"
            else:
                md += f"### Table on Page {table['page']}\n\n"
            
            if 'headers' in table and table['headers']:
                headers = [str(h).strip() if h is not None else '' for h in table['headers']]
                content = [[str(cell).strip() if cell is not None else '' for cell in row] for row in table['content']]
                
                # Ensure the number of headers matches the number of columns in the content
                max_columns = max(len(row) for row in content)
                headers = headers[:max_columns]  # Truncate headers if there are too many
                headers += [''] * (max_columns - len(headers))  # Add empty headers if there are too few
                
                df = pd.DataFrame(content, columns=headers)
            else:
                content = [[str(cell).strip() if cell is not None else '' for cell in row] for row in table['content']]
                df = pd.DataFrame(content)
            
            # Generate markdown table
            table_md = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
            
            # Remove all spaces around pipe symbols, including the leading and trailing pipes
            table_md = '\n'.join(['|' + '|'.join([cell.strip() for cell in row.split('|')[1:-1]]) + '|' for row in table_md.split('\n')])
            
            md += table_md + "\n\n"
        
        return md

def save_sales_report_to_db(extracted_content: Dict[str, Any], db_manager: DBManager):
    sales_data = []
    for table in extracted_content['tables']:
        if 'headers' in table and table['headers']:
            if any('sales' in header.lower() for header in table['headers'] if header):
                process_table_with_headers(table, sales_data)
        else:
            if table['content'] and len(table['content'][0]) >= 6:
                process_table_without_headers(table, sales_data)
    
    if sales_data:
        db_manager.create_sales_report_table()
        db_manager.insert_sales_report_data(sales_data)
        logger.info(f"Sales Report data saved to database. Rows inserted: {len(sales_data)}")
    else:
        logger.warning("No Sales Report data found in the extracted content")

def process_table_with_headers(table: Dict[str, Any], sales_data: List[Dict[str, str]]):
    for row in table['content']:
        try:
            sales_row = create_sales_row(row)
            sales_data.append(sales_row)
        except Exception as e:
            logger.error(f"Error processing row with headers: {row}. Error: {str(e)}")

def process_table_without_headers(table: Dict[str, Any], sales_data: List[Dict[str, str]]):
    for row in table['content']:
        try:
            sales_row = create_sales_row(row)
            sales_data.append(sales_row)
        except Exception as e:
            logger.error(f"Error processing row without headers: {row}. Error: {str(e)}")

def create_sales_row(row: List[str]) -> Dict[str, str]:
    return {
        'Month': row[0],
        'Electronics': row[1] if len(row) > 1 else '',
        'Clothing': row[2] if len(row) > 2 else '',
        'Food': row[3] if len(row) > 3 else '',
        'Books': row[4] if len(row) > 4 else '',
        'Total': row[-1]
    }

# Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    pdf_path = "complex_sample.pdf"  # Replace with your PDF path
    extractor = PDFExtractor(pdf_path)
    extractor.extract_content()
    
    # Log extracted content for debugging
    logger.debug(f"Extracted tables: {json.dumps(extractor.extracted_data['tables'], indent=2, ensure_ascii=False)}")

    # Save extracted content as markdown
    with open("output.md", "w", encoding="utf-8") as f:
        f.write(extractor.to_markdown())

    # Save Sales Report to database
    save_sales_report_to_db(extractor.extracted_data, extractor.db_manager)

    # Display all entries in the sales_report table
    all_sales_data = extractor.db_manager.get_all_sales_report_data()
    logger.info(f"All entries in the sales_report table (count: {len(all_sales_data)}):")
    for row in all_sales_data:
        logger.info(json.dumps(row, indent=2, ensure_ascii=False))