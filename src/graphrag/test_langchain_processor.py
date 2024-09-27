import unittest
from src.graphrag.langchain_text_processor import process_text_files
from src.graphrag.langchain_ai_service import get_langchain_ai_service
import yaml

class TestLangChainProcessor(unittest.TestCase):
    def setUp(self):
        with open("config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        self.config['ai_service'] = 'mock'
        self.ai_service = get_langchain_ai_service(self.config['ai_service'])

    def test_process_text_files(self):
        # Create a temporary folder with a test file
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdirname:
            # Create a test file
            with open(os.path.join(tmpdirname, "test.txt"), "w") as f:
                f.write("This is a test document.")

            # Process the files
            process_text_files(tmpdirname)

            # Add assertions here to check if the processing worked as expected
            # For example, you could check if certain methods of your database manager were called

if __name__ == '__main__':
    unittest.main()