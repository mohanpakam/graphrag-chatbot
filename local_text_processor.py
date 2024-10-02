from local_llama_processor import LocalLlamaProcessor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting the Local Text Processor...")
    
    processor = LocalLlamaProcessor()
    
    # Sample text to process
    sample_text = """
    On 27th of September 2023, The devastation in North Carolina in the wake of Hurricane Helene could have serious implications for a niche — but extremely important — corner of the tech industry.

    Tucked in the Blue Ridge Mountains on the outskirts of Spruce Pine, a town of less than 2,200, are two mines that produce the world's purest quartz, which formed in the area some 380 million years ago. The material is a key component in the global supply chain for semiconductor chips, which power everything from smartphones and cars to medical devices and solar panels.

    But operations at the facilities have halted since Hurricane Helene tore through the southeast United States over the weekend, causing historic flooding and landslides, cutting off roads and power and endangering millions of residents.
    """
    
    # Process the sample text
    processor.process_text(sample_text, "sample_document.txt")
    
    logger.info("Sample text processed. Starting user query loop...")
    
    while True:
        query = input("Enter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        
        response = processor.graph_rag.process_query(query)
        
        logger.info(f"Query: {query}")
        logger.info(f"Response: {response}")
        print(f"\nResponse: {response}\n")
    
    logger.info("Local Text Processor completed.")

if __name__ == "__main__":
    main()