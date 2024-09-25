# AI Chatbot with Streamlit and FastAPI

This project implements an AI chatbot using Streamlit for the frontend and FastAPI for the backend, with PDF processing and vector search capabilities.

## Prerequisites

- Python 3.7 or higher
- SQLite3 with vector extension

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ai-chatbot.git
   cd ai-chatbot
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Install SQLite3 and enable vector extension:

   a. For Ubuntu/Debian:
   ```bash
   sudo apt-get update
   sudo apt-get install sqlite3 libsqlite3-dev
   ```

   b. For macOS (using Homebrew):
   ```bash
   brew install sqlite
   ```

   c. For Windows:
   Download the SQLite precompiled binaries for Windows from the official SQLite website: https://www.sqlite.org/download.html

   After installation, you need to compile the SQLite VSS extension:

   ```bash
   git clone https://github.com/asg017/sqlite-vss
   cd sqlite-vss
   make
   ```

   This will create a `vss0.so` (on Unix-based systems) or `vss0.dll` (on Windows) file. Note the path to this file.

5. Update the `config.yaml` file with your API keys, file paths, and other settings.

## Running the Application

1. Process PDFs and create the vector database:
   ```bash
   python pdf_processor.py
   ```

2. Start the FastAPI backend:
   ```bash
   python api.py
   ```
   The API will be available at `http://localhost:8000`.

3. In a new terminal, start the Streamlit frontend:
   ```bash
   streamlit run app.py
   ```
   The Streamlit app will open in your default web browser.

## Usage

1. Place your PDF files in the `pdfs` folder (or the folder specified in `config.yaml`).
2. Run the PDF processor to create the vector database.
3. Start the backend and frontend applications.
4. Enter your message in the text input field on the Streamlit interface.
5. Click the "Send" button or press Enter to submit your message.
6. The AI's response will appear in the chat history.
7. Use the "Clear Chat" button to reset the conversation.

## Project Structure

- `app.py`: Streamlit frontend application
- `api.py`: FastAPI backend server
- `pdf_processor.py`: Script to process PDFs and create vector embeddings
- `config.yaml`: Configuration file for API keys, file paths, and other settings
- `requirements.txt`: List of Python dependencies

## Notes

- Ensure that the SQLite vector extension is properly installed and accessible.
- The current implementation uses Azure OpenAI. Make sure to update the `config.yaml` with your Azure OpenAI API key and endpoint.
- This is a basic implementation. In a production environment, you would need to implement proper error handling, security measures, and optimize for performance.

## Troubleshooting

If you encounter issues with the SQLite vector extension:

1. Ensure that the `sqlite_vss` extension is in a location accessible by SQLite.
2. You may need to update the `load_extension()` call in `api.py` with the full path to the extension file:

   ```python
   conn.load_extension("/path/to/vss0.so")  # or vss0.dll on Windows
   ```

3. On some systems, you might need to set the `LD_LIBRARY_PATH` environment variable to include the directory containing the SQLite libraries:

   ```bash
   export LD_LIBRARY_PATH=/path/to/sqlite/lib:$LD_LIBRARY_PATH
   ```

## License

This project is open source and available under the [MIT License](LICENSE).

## Local Text Processing with Ollama

This project now includes an option for local text processing using Ollama with the Llama2 model:

1. Install Ollama:
   - For macOS or Linux:
     ```bash
     curl https://ollama.ai/install.sh | sh
     ```
   - For Windows:
     Download the installer from [Ollama's official website](https://ollama.ai/download)

2. Install the Llama2 model:
   ```bash
   ollama pull llama2
   ```

3. Start the Ollama server:
   ```bash
   ollama serve
   ```

4. Update the `config.yaml` file to include the Ollama API endpoint:
   ```yaml
   # Ollama Configuration
   ollama_api_url: "http://localhost:11434/api/embeddings"
   ollama_model: "llama2"
   ```

5. Run the local text processor:
   ```bash
   python local_text_processor.py
   ```

6. This script will process text files from the `text_folder` specified in `config.yaml`, create embeddings using the local Ollama API, and store them in the SQLite database.

7. The script will display real-time processing information and final statistics about API calls and response times.

Note: Ensure that the Ollama server is running before executing the `local_text_processor.py` script.