# Configuration and fixed response templates for the deployed airport chatbot
from dataclasses import dataclass


# Define the immutable settings container used by the deployed application
@dataclass(frozen=True)
class AirportConfig:
    # Vision-language model that embeds images and short texts into one shared space
    clip_model_name: str = "openai/clip-vit-base-patch32"
    # Sentence embedding model used for full natural-language passenger questions
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Size of the Whisper speech-to-text model
    speech_model_name: str = "base"
    # Square pixel size that all images are resized to before encoding
    image_size: int = 224
    # Number of knowledge base records returned by a retrieval call
    top_k: int = 3
    # Minimum raw cosine similarity below which the chatbot refuses a text query
    text_confidence_threshold: float = 0.35
    # Minimum raw cosine similarity below which the chatbot refuses an image query
    image_confidence_threshold: float = 0.22
    # Relative contribution of the image evidence during multimodal fusion
    image_weight: float = 0.5
    # Relative contribution of the text evidence during multimodal fusion
    text_weight: float = 0.5
    # File name of the structured airport knowledge base
    kb_path: str = "airport_kb.json"


# Create the single configuration instance used across the application
CONFIG = AirportConfig()

# Fixed response returned whenever the system is not confident enough to answer
UNCERTAIN_RESPONSE = (
    "I am not confident enough to answer that reliably. "
    "Please check the airport information screens or speak to a member of staff at the nearest "
    "information desk."
)

# Fixed reminder attached to any answer involving time-sensitive information
LIVE_DATA_DISCLAIMER = (
    "Gate, delay and opening-hour information can change at short notice. "
    "Always confirm against the live airport departure boards."
)

# Notice shown to the passenger before any personal data is uploaded
PRIVACY_NOTICE = (
    "Images and audio are processed in memory on this server and are not stored, logged or "
    "transmitted to any third party. Please avoid uploading your boarding pass, which contains "
    "your name and booking reference."
)
