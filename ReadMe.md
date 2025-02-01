# MediAssist: AI Healthcare Assistant 🎙️

MediAssist is an interactive voice-enabled AI healthcare assistant that provides medical information and guidance through both text and voice interactions. Built with state-of-the-art language models and speech processing capabilities, it offers a natural way to access healthcare information while maintaining appropriate medical guidance boundaries.

## Features

- 🗣️ Voice Recognition & Text-to-Speech
- 💬 Natural Language Processing
- 📚 RAG-based Knowledge Retrieval
- 🏥 Healthcare-focused Responses
- 🖥️ Interactive Streamlit Interface

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for optimal performance)
- Anaconda or Miniconda

## Installation

1. **Install Anaconda/Miniconda**

   Download and install Anaconda or Miniconda from:

   - Anaconda: [https://www.anaconda.com/download](https://www.anaconda.com/download)
   - Miniconda: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

2. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/mediassist.git
   cd mediassist
   ```

3. **Create and Activate Conda Environment**

   ```bash
   conda env create --name mediassist -f environment.yml
   ```

4. **Install Ollama**

   Follow the installation instructions for your operating system:

   - For macOS/Linux: [https://ollama.ai/download](https://ollama.ai/download)
   - For Windows: Use WSL2 to install Ollama

5. **Download Required Models**
   ```bash
   ollama pull llama3.2:1b
   ollama pull nomic-embed-text
   ```

## Configuration

1. **Prepare Knowledge Base**

   - Place your medical knowledge base PDF in the project root directory
   - Update the PDF path in `AIVoiceAssistant.py` if needed:
     ```python
     pdf_path="./your_knowledge_base.pdf"
     ```

2. **Audio Setup**
   - Ensure your microphone is properly connected and configured
   - Test your system's audio input/output capabilities

## Usage

1. **Start the Application**

   ```bash
   streamlit run app.py
   ```

2. **Interact with MediAssist**
   - Type questions in the chat input
   - Click the microphone button for voice input
   - Receive both text and voice responses

## Project Structure

```
mediassist/
├── app.py                 # Main Streamlit application
├── pipeline.py           # RAG pipeline implementation
├── voice_service.py      # Voice processing services
├── AIVoiceAssistant.py   # Core assistant logic
├── environment.yml       # Conda environment specification
└── README.md            # Project documentation
```

## Dependencies

Key dependencies include:

- streamlit
- faster-whisper
- pyaudio
- faiss-cpu
- ollama
- pytorch
- gtts
- pygame
- pydub
- spacy
- PyMuPDF

All dependencies are managed through the conda environment.

## Note on Medical Advice

This AI assistant is designed to provide general medical information and guidance. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers for medical concerns.

## Troubleshooting

Common issues and solutions:

1. **Audio Device Not Found**

   - Check system audio settings
   - Ensure PyAudio is properly installed
   - Try reinstalling audio dependencies

2. **CUDA/GPU Issues**

   - Verify CUDA installation
   - Update GPU drivers
   - Check CUDA compatibility with PyTorch

3. **Model Loading Errors**
   - Ensure Ollama is running
   - Verify model downloads
   - Check available disk space

For more issues, please check the GitHub issues section or create a new issue.
