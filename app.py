import os
import wave
import pyaudio
import numpy as np
import streamlit as st
import requests
from datetime import datetime
import logging
import nest_asyncio
import coloredlogs
from scipy.io import wavfile
from faster_whisper import WhisperModel
from AIVoiceAssistant import AIVoiceAssistant
import voice_service as vs

# Initialize async support
nest_asyncio.apply()

# Hide Pygame support prompt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

# Configure logging with colors
coloredlogs.install(
    level="INFO",
    fmt="%(asctime)s - %(levelname)s - %(message)s",
    field_styles={
        "asctime": {"color": "blue"},
        "levelname": {"bold": True, "color": "black"},
        "message": {"color": "white"},
    },
    level_styles={
        "info": {"color": "green"},
        "warning": {"color": "yellow"},
        "error": {"color": "red", "bold": True},
    },
)

# Constants
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
DEFAULT_MODEL_SIZE = "turbo"
DEFAULT_CHUNK_LENGTH = 10
SILENCE_THRESHOLD = 3000
BACKEND_URL = "http://localhost:8000"


# Add this new function before the VoiceAssistantApp class
def init_whisper_model(
    model_size=DEFAULT_MODEL_SIZE, device="cuda", compute_type="float16", num_workers=10
):
    """Initialize Whisper model with proper error handling and progress bar management."""
    # Temporarily disable tqdm progress bars
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    try:
        logging.info(f"Loading Whisper model: {model_size}")
        model = WhisperModel(
            f"{model_size}",
            device=device,
            compute_type=compute_type,
            num_workers=num_workers,
            download_root="./models",  # Local cache directory
        )
        logging.info("Whisper model loaded successfully")
        return model
    except Exception as e:
        logging.error(f"Failed to load Whisper model: {str(e)}")
        raise


# Modified VoiceAssistantApp class
class VoiceAssistantApp:
    def __init__(self):
        logging.info("Initializing Voice Assistant...")
        try:
            # Initialize Whisper model with error handling
            self.model = init_whisper_model()

            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
            )
            self.ai_assistant = AIVoiceAssistant()

        except Exception as e:
            logging.error(f"Error initializing Voice Assistant: {str(e)}")
            # Clean up resources if initialization fails
            if hasattr(self, "stream"):
                self.stream.close()
            if hasattr(self, "audio"):
                self.audio.terminate()
            raise

    def is_silence(self, data, threshold=SILENCE_THRESHOLD):
        """Check if the audio data contains silence."""
        max_amplitude = np.max(np.abs(data))
        return max_amplitude <= threshold

    def record_audio_chunk(self):
        """Record an audio chunk and return whether it's silent or not."""
        frames = []
        for _ in range(0, int(SAMPLE_RATE / CHUNK_SIZE * DEFAULT_CHUNK_LENGTH)):
            data = self.stream.read(CHUNK_SIZE)
            frames.append(data)

        temp_file = "temp_audio_chunk.wav"
        with wave.open(temp_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))

        try:
            _, data = wavfile.read(temp_file)
            if self.is_silence(data):
                os.remove(temp_file)
                logging.warning("Silence detected, skipping...")
                return True, None
            return False, temp_file
        except Exception as e:
            logging.error(f"Error processing audio: {e}")
            return True, None

    def transcribe_audio(self, file_path):
        """Transcribe recorded audio using Whisper."""
        try:
            segments, _ = self.model.transcribe(file_path, beam_size=7)
            return " ".join(segment.text for segment in segments)
        except Exception as e:
            logging.error(f"Error transcribing audio: {e}")
            return ""

    def cleanup(self):
        """Clean up resources."""
        logging.info("Cleaning up resources...")
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        logging.info("Voice Assistant stopped.")


# Modified init_session_state function
def init_session_state():
    """Initialize session state with proper error handling."""
    try:
        if "logged_in" not in st.session_state:
            st.session_state["logged_in"] = False
        if "name" not in st.session_state:
            st.session_state["name"] = None
        if "email" not in st.session_state:
            st.session_state["email"] = None
        if "user_id" not in st.session_state:
            st.session_state["user_id"] = None
        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {
                    "role": "assistant",
                    "content": "👋Hello! I'm your AI healthcare assistant. I can help you with medical queries and provide information based on verified medical guidelines. Please note that I'm not a replacement for professional medical advice. How can I assist you today?",
                }
            ]

        # Initialize voice assistant with error handling
        if "voice_assistant" not in st.session_state:
            try:
                st.session_state["voice_assistant"] = VoiceAssistantApp()
            except Exception as e:
                st.error(f"Failed to initialize voice assistant: {str(e)}")
                # Provide fallback behavior
                st.session_state["voice_assistant"] = None
                logging.error(f"Voice assistant initialization failed: {str(e)}")

    except Exception as e:
        logging.error(f"Error in init_session_state: {str(e)}")
        st.error(
            "An error occurred while initializing the application. Please try refreshing the page."
        )


def apply_custom_styles():
    st.markdown(
        """
        <style>
        /* Existing styles */
        .stApp {
            background-color: #323946;
        }
        .stApp header {
            background-color: #323946;
        }
        .stApp footer {
            background-color: #323946;
        }
        .st-emotion-cache-hzygls{
            background-color: #323946;
        }
        [data-testid="stSidebar"] {
            background-color: #1F242D;
            box-shadow: 1px 0 10px rgba(0, 255, 255, 0.3),
                2px 0 20px rgba(0, 255, 255, 0.2),
                3px 0 30px rgba(0, 255, 255, 0.1);
        }
        [data-testid="stSidebar"] * {
            color: white;
        }
        [data-testid="stChatInput"] {
            background-color: #ffffff;
        }
        [data-testid="stChatInput"] {
            background-color: #1F242D;
            color: #00EEFF;
        }
        /* Change the placeholder text color */
        [data-testid="stChatInput"] textarea::placeholder {
            color: #FFF;
        }
        /* Change the send button */
        [data-testid="stChatInput"] button {
            color: #00EEFF;
        }
        /* Center buttons in the sidebar */
        [data-testid="stSidebar"] .stButton {
            margin-top: 15px;
            display: flex;
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Main application
st.set_page_config(page_title="MediAssist", page_icon="🎙️")
init_session_state()
apply_custom_styles()

# Sidebar UI
# Sidebar UI with AI Model Information
st.sidebar.markdown(
    "<h1 style='text-align: center;'>MediAssist</h1>", unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div class='sidebar-info'>
        <h4>🤖 Model Information</h4>
        <hr>
        <p>🔊 Speech Model: Whisper Medium</p>
        <p>🧠 LLM: llama3.2 (1B params)</p>
        <p>📚 Embeddings: Nomic Embed</p>
        <p>🎯 Temperature: 0.7</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <div class='sidebar-info'>
        <h4>🔍 System Capabilities</h4>
        <hr>
        <p>🗣️ Voice-to-Text Transcription</p>
        <p>📝 Context-Aware Responses</p>
        <p>😊 Sentiment Analysis</p>
        <p>🔊 Text-to-Speech Output</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <div class='sidebar-info'>
        <h4>⚠️ Important Notice</h4>
        <hr>
        <p>This AI assistant provides general medical information only. Always consult healthcare professionals for medical advice.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# Main chat interface
st.header("🎙️ MediAssist : AI Healthcare Assistant")

# Display chat messages
for message in st.session_state["messages"]:
    with st.chat_message(
        message["role"], avatar="🤖" if message["role"] == "assistant" else "🗣️"
    ):
        st.write(message["content"])

# Handle text input
if prompt := st.chat_input("Type your medical question here..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        response_placeholder = st.empty()
        response = ""

        with st.spinner("🩺 Thinking..."):
            response_generator = st.session_state[
                "voice_assistant"
            ].ai_assistant.interact_with_llm(prompt)

            for chunk in response_generator:
                response += chunk
                response_placeholder.markdown(response + "▌")

        response_placeholder.markdown(response)

    st.session_state["messages"].append({"role": "assistant", "content": response})

# Handle voice input
# Handle voice input
col1, col2 = st.columns([6, 1])
with col1:
    if st.button("🎙️ Record"):
        # Indicate "Listening..."
        with st.spinner("Listening..."):
            is_silent, audio_file = st.session_state[
                "voice_assistant"
            ].record_audio_chunk()

            if not is_silent and audio_file:
                transcription = st.session_state["voice_assistant"].transcribe_audio(
                    audio_file
                )
                os.remove(audio_file)

                if transcription.strip():
                    st.session_state["messages"].append(
                        {"role": "user", "content": transcription}
                    )
                    st.chat_message("user").write(transcription)
                    with st.spinner("Thinking..."):
                        # Indicate "Thinking..." during LLM response generation
                        with st.chat_message("assistant", avatar="🤖"):
                            response_placeholder = st.empty()
                            response = ""

                            response_generator = st.session_state[
                                "voice_assistant"
                            ].ai_assistant.interact_with_llm(transcription)

                            for chunk in response_generator:
                                response += chunk
                                response_placeholder.markdown(response + "▌")

                            response_placeholder.markdown(response)

                        st.session_state["messages"].append(
                            {"role": "assistant", "content": response}
                        )

                    with st.spinner("Reading..."):
                        vs.play_text_to_speech(response)
