from pdf_extractor import PDFExtractor, measure_effectiveness, process_with_llm, save_sales_report_to_db
from create_sample_pdf import create_sample_pdf
import json
import logging

logger = logging.getLogger(__name__)

def test_extraction():
    # Create the complex sample PDF
    create_sample_pdf("complex_sample.pdf")
    
    # Extract content from the PDF
    extractor = PDFExtractor("complex_sample.pdf")
    extractor.extract_content()
    
    # Save extracted content in different formats
    with open("output.json", "w") as f:
        f.write(extractor.to_json())
    
    with open("output.html", "w") as f:
        f.write(extractor.to_html())
    
    with open("output.md", "w") as f:
        f.write(extractor.to_markdown())
    
    # Measure effectiveness
    effectiveness = measure_effectiveness("complex_sample.pdf", extractor.extracted_data)
    logger.info(f"Extraction effectiveness: {effectiveness:.2f}%")
    
    # Process with LLM
    llm_processed_content = process_with_llm(extractor.extracted_data)
    
    # Save LLM processed content
    with open("llm_output.json", "w", encoding='utf-8') as f:
        json.dump(llm_processed_content, f, indent=2, ensure_ascii=False)

    logger.info("LLM processing complete. Results saved in llm_output.json")
    
    # Save Sales Report to database
    save_sales_report_to_db(llm_processed_content, extractor.db_manager)

    # Display all entries in the sales_report table
    all_sales_data = extractor.db_manager.get_all_sales_report_data()
    logger.info("All entries in the sales_report table:")
    for row in all_sales_data:
        logger.info(json.dumps(row, indent=2))
    
    return effectiveness, llm_processed_content

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    effectiveness, llm_processed_content = test_extraction()
    
    if effectiveness >= 90:
        logger.info("Extraction effectiveness is satisfactory.")
    else:
        logger.warning("Further improvements to the extraction process are needed.")

    logger.info("\nLLM Processed Content:")
    logger.info(f"Summary: {llm_processed_content['summary'][:500]}...")  # Log first 500 characters
    logger.info("\nKey Points:")
    for point in llm_processed_content['key_points']:
        logger.info(f"- {point}")
    logger.info(f"\nInsights: {llm_processed_content['insights'][:500]}...")  # Log first 500 characters
    logger.info("\nTables:")
    for table in llm_processed_content['tables']:
        logger.info(f"Table Name: {table.get('table_name', 'N/A')}")
        logger.info(f"Description: {table.get('description', 'N/A')}")
        logger.info("Data:")
        if table.get('table_name') == "Sales Report":
            for row in table.get('data', [])[:5]:  # Print first 5 rows for brevity
                logger.info(json.dumps(row, indent=2))
        else:
            logger.info(json.dumps(table.get('data', []), indent=2))
        logger.info("")

    if not any(llm_processed_content.values()):
        logger.error("LLM processing failed to produce any content. Check the LLM service and prompt.")