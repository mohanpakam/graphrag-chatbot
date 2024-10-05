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
            'tables': [],
            'images': []
        }
        self.db_manager = DBManager('sales_report.db')

    def extract_content(self):
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            self.extract_text(page, page_num)
            self.extract_tables(page, page_num)
            self.extract_images(page, page_num)

    def extract_text(self, page, page_num):
        text = page.get_text()
        self.extracted_data['text'].append({
            'page': page_num + 1,
            'content': text
        })

    def extract_tables(self, page, page_num):
        tables = page.find_tables()
        for table in tables.tables:
            extracted_table = table.extract()
            if extracted_table:
                headers = extracted_table[0]
                data = extracted_table[1:]
                df = pd.DataFrame(data, columns=headers)
                self.extracted_data['tables'].append({
                    'page': page_num + 1,
                    'headers': headers,
                    'content': df.to_dict('records')
                })
        logger.info(f"Extracted {len(tables.tables)} tables from page {page_num + 1}")

    def extract_images(self, page, page_num):
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = self.doc.extract_image(xref)
            image_bytes = base_image["image"]
            self.extracted_data['images'].append({
                'page': page_num + 1,
                'index': img_index,
                'bytes': image_bytes
            })

    def to_json(self):
        return json.dumps(self.extracted_data, indent=2, default=str)

    def to_html(self):
        html = "<html><body>"
        for text in self.extracted_data['text']:
            html += f"<h2>Page {text['page']}</h2>"
            html += f"<p>{text['content']}</p>"
        
        for table in self.extracted_data['tables']:
            html += f"<h3>Table on Page {table['page']}</h3>"
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
            md += f"### Table on Page {table['page']}\n\n"
            df = pd.DataFrame(table['content'])
            md += tabulate(df, headers='keys', tablefmt='pipe') + "\n\n"
        
        return md

def save_sales_report_to_db(extracted_content: Dict[str, Any], db_manager: DBManager):
    sales_data = []
    for table in extracted_content['tables']:
        if table['page'] in [1, 2]:  # Only consider tables from pages 1 and 2
            # Log the structure of the table content
            logger.debug(f"Table content structure: {json.dumps(table['content'][0] if table['content'] else {}, indent=2)}")
            
            for row in table['content']:
                try:
                    sales_row = {
                        'Month': row.get('Month', row.get('0', '')),  # Assume 'Month' might be labeled as '0'
                        'Electronics': row.get('Electronics', row.get('1', '')),
                        'Clothing': row.get('Clothing', row.get('2', '')),
                        'Food': row.get('Food', row.get('3', '')),
                        'Books': row.get('Books', row.get('4', '')),
                        'Total': row.get('Total', row.get('5', ''))
                    }
                    sales_data.append(sales_row)
                except KeyError as e:
                    logger.error(f"KeyError when processing row: {row}. Error: {str(e)}")
                except Exception as e:
                    logger.error(f"Unexpected error when processing row: {row}. Error: {str(e)}")
    
    if sales_data:
        db_manager.create_sales_report_table()
        db_manager.insert_sales_report_data(sales_data)
        logger.info(f"Sales Report data saved to database. Rows inserted: {len(sales_data)}")
    else:
        logger.warning("No Sales Report data found on pages 1 or 2")

# Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    pdf_path = "complex_sample.pdf"  # Replace with your PDF path
    extractor = PDFExtractor(pdf_path)
    extractor.extract_content()
    
    # Log extracted content for debugging
    logger.debug(f"Extracted text: {extractor.extracted_data['text']}")
    logger.debug(f"Extracted tables: {extractor.extracted_data['tables']}")

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