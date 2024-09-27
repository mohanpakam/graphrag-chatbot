import unittest
from unittest.mock import Mock, patch
from langchain.schema import Document as LangChainDocument
from langchain_text_processor import convert_chunks_to_graph_documents

class TestLangChainProcessor(unittest.TestCase):

    @patch('langchain_text_processor.LLMGraphTransformer')
    @patch('langchain_text_processor.logger')
    def test_convert_chunks_to_graph_documents(self, mock_logger, MockLLMGraphTransformer):
        # Create mock chunks
        chunks = [
            LangChainDocument(page_content="Chunk 1", metadata={"source": "file1.txt"}),
            LangChainDocument(page_content="Chunk 2", metadata={"source": "file1.txt"}),
            LangChainDocument(page_content="Chunk 3", metadata={"source": "file1.txt"}),
        ]

        # Create mock LLMGraphTransformer
        mock_transformer = MockLLMGraphTransformer.return_value
        mock_transformer.convert_to_graph_documents.return_value = [
            Mock(source=chunk) for chunk in chunks
        ]

        # Call the function
        filename = "test_file.txt"
        result = convert_chunks_to_graph_documents(chunks, mock_transformer, filename)

        # Assertions
        self.assertEqual(len(result), 3)
        mock_transformer.convert_to_graph_documents.assert_called_once_with(chunks)

        # Check if logger.info was called with the correct message
        mock_logger.info.assert_called_with(f"Converted 3 chunks to graph documents for {filename}")

    @patch('langchain_text_processor.LLMGraphTransformer')
    @patch('langchain_text_processor.logger')
    def test_convert_chunks_to_graph_documents_with_error(self, mock_logger, MockLLMGraphTransformer):
        # Create mock chunks
        chunks = [
            LangChainDocument(page_content="Chunk 1", metadata={"source": "file1.txt"}),
            LangChainDocument(page_content="Chunk 2", metadata={"source": "file1.txt"}),
            LangChainDocument(page_content="Chunk 3", metadata={"source": "file1.txt"}),
        ]

        # Create mock LLMGraphTransformer
        mock_transformer = MockLLMGraphTransformer.return_value
        mock_transformer.convert_to_graph_documents.side_effect = Exception("Test error")

        # Call the function
        filename = "test_file.txt"
        result = convert_chunks_to_graph_documents(chunks, mock_transformer, filename)

        # Assertions
        self.assertEqual(len(result), 0)  # No successful conversions
        mock_transformer.convert_to_graph_documents.assert_called_once_with(chunks)

        # Check if logger.error was called for the exception
        mock_logger.error.assert_called()
        mock_logger.error.assert_any_call("Error processing chunk 0 in test_file.txt: Test error")

        # Check if logger.info was called with the correct message
        mock_logger.info.assert_called_with(f"Converted 0 chunks to graph documents for {filename}")

if __name__ == '__main__':
    unittest.main()