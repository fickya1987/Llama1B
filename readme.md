# Small Language Models - Proof of Concept

This project demonstrates a Streamlit application for interacting with various small language models via the Hugging Face API. The application allows users to chat with different models and upload documents for question answering.

## Features

- Chat with multiple language models (Llama 3.2, Phi-3.5, Gemma, DeepSeek)
- Upload and process documents (PDF, DOCX, TXT)
- Document-based question answering
- Adjustable model parameters (temperature, top-p, max length)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Hugging Face API token

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/slm-poc.git
   cd slm-poc
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your Hugging Face API token:
   ```
   HF_TOKEN=hf_your_token_here
   ```

### Running the Application

```
streamlit run app.py
```

## Usage

1. Select a model from the dropdown in the sidebar
2. Adjust model parameters if needed
3. Start chatting with the model
4. Optionally upload a document for document-based question answering

## Project Structure

- `app.py`: Main Streamlit application
- `config/`: Configuration settings
- `models/`: Model implementations
- `utils/`: Utility functions
- `components/`: UI components

## Adding New Models

To add a new model:

1. Add the model details to `config/settings.py`
2. Ensure the model is compatible with the Hugging Face API

## License

This project is licensed under the MIT License - see the LICENSE file for details.