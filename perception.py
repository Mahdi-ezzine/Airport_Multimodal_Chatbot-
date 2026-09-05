# Model loading and encoding functions for the deployed application
import os
import re
import time
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from airport_config import CONFIG

# Determine whether the deployed application runs on GPU or CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load every frozen model once and return them together
def load_models():
    # Load the CLIP processor, requesting the fast image processor where it is supported
    try:
        clip_processor = CLIPProcessor.from_pretrained(CONFIG.clip_model_name, use_fast=True)
    except TypeError:
        clip_processor = CLIPProcessor.from_pretrained(CONFIG.clip_model_name)
    clip_model = CLIPModel.from_pretrained(CONFIG.clip_model_name).to(DEVICE)
    # Disable dropout for inference
    clip_model.eval()

    # Load the sentence embedding model
    text_model = SentenceTransformer(CONFIG.text_model_name, device=str(DEVICE))

    # Load Whisper lazily, since it is only needed when audio is supplied
    import whisper
    speech_model = whisper.load_model(CONFIG.speech_model_name, device=str(DEVICE))

    # Return all three models to the caller
    return clip_processor, clip_model, text_model, speech_model


# Build the encoding and transcription functions bound to the loaded models
def build_encoders(clip_processor, clip_model, text_model, speech_model):

    # Extract a projected CLIP embedding whatever the Transformers version returns
    def clip_features_from(output, projection):
        # Measure the projection dimensions rather than assuming what the model returned
          ## pooler_output is the raw hidden state in some releases and the already-projected
          ## embedding in others, so the width is checked before deciding whether to project
        input_dimension, output_dimension = projection.in_features, projection.out_features

        # Decide whether a candidate tensor still needs projecting
        def finalise(tensor):
            if tensor.shape[-1] == output_dimension:
                return tensor
            if tensor.shape[-1] == input_dimension:
                return projection(tensor)
            raise ValueError(
                f"CLIP returned a tensor of width {tensor.shape[-1]}, matching neither the "
                f"projection input ({input_dimension}) nor its output ({output_dimension})."
            )

        # Older releases return the embedding directly as a tensor
        if isinstance(output, torch.Tensor):
            return finalise(output)
        # Some releases return an object carrying the embedding under a named field
        for attribute in ("image_embeds", "text_embeds", "pooler_output"):
            value = getattr(output, attribute, None)
            if isinstance(value, torch.Tensor):
                return finalise(value)
        # Some releases return a plain tuple whose first element is the embedding
        if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
            return finalise(output[0])
        # Fail clearly rather than returning something that would corrupt retrieval
        raise TypeError(f"Unexpected CLIP output type: {type(output)}")

    # Embed one or more images into CLIP space
    def encode_images(images: Sequence[Any]) -> np.ndarray:
        # Load every source into an RGB PIL image
        pil_images = [
            source if isinstance(source, Image.Image) else Image.open(source)
            for source in images
        ]
        pil_images = [image.convert("RGB") for image in pil_images]
        # Let the CLIP processor handle resizing and channel normalisation
        inputs = clip_processor(images=pil_images, return_tensors="pt").to(DEVICE)
        # Run the encoder and the projection inside one inference_mode block
          ## A tensor created in inference mode cannot be fed to an autograd-tracked
          ## layer from outside the block, so the projection must happen here
        with torch.inference_mode():
            output = clip_model.get_image_features(**inputs)
            features = clip_features_from(output, clip_model.visual_projection)
            features = features / features.norm(dim=-1, keepdim=True)
        # Return the result as a float32 array for FAISS
        return features.cpu().numpy().astype("float32")

    # Embed text into CLIP space
    def encode_text_clip(texts: Sequence[str]) -> np.ndarray:
        # Tokenise the texts within CLIP's 77-token limit
        inputs = clip_processor(
            text=list(texts), return_tensors="pt", padding=True, truncation=True, max_length=77
        ).to(DEVICE)
        # Run the encoder and the projection inside one inference_mode block
        with torch.inference_mode():
            output = clip_model.get_text_features(**inputs)
            features = clip_features_from(output, clip_model.text_projection)
            features = features / features.norm(dim=-1, keepdim=True)
        # Return the result as a float32 array for FAISS
        return features.cpu().numpy().astype("float32")

    # Embed text into sentence-transformer space
    def encode_text_semantic(texts: Sequence[str]) -> np.ndarray:
        # Encode the texts with normalisation enabled
        encoded = text_model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        # Return the result as a float32 array for FAISS
        return np.asarray(encoded, dtype="float32")

    # Transcribe one audio file into text
    def transcribe_audio(path: str) -> Dict[str, Any]:
        # Fail clearly if the file does not exist
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        # Record the start time so latency can be reported
        started_at = time.time()
        # Run Whisper with the language fixed to English
        result = speech_model.transcribe(path, language="en", fp16=(DEVICE.type == "cuda"))
        # Return the transcript and its latency
        return {
            "transcript": str(result.get("text", "")).strip(),
            "seconds_elapsed": round(time.time() - started_at, 2),
        }

    # Return every function to the caller
    return encode_images, encode_text_clip, encode_text_semantic, transcribe_audio


# Spoken number words, converted to digits before entity extraction
  ## Whisper writes gate identifiers as "b twelve" or "b 12" rather than "B12", and the gate
  ## number is the entity a passenger most needs resolved correctly
SPOKEN_NUMBERS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
    "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15", "sixteen": "16",
    "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
}


# Convert spoken number words into digits and rejoin split identifiers
def normalise_spoken_numbers(text: str) -> str:
    for word, digit in SPOKEN_NUMBERS.items():
        text = re.sub(rf"\b{word}\b", digit, text)
    return re.sub(r"\b([a-z])[\s\-]+(\d{1,2})\b", r"\1\2", text)


# Airport-specific abbreviations expanded before embedding
AIRPORT_ABBREVIATIONS = {
    r"\bt(\d)\b": r"terminal \1",
    r"\bgt\b": "gate",
    r"\bbag claim\b": "baggage claim",
    r"\bwc\b": "restroom",
    r"\btoilets?\b": "restroom",
    r"\binfo\b": "information",
}

# Regular expressions for the entities the chatbot recognises
ENTITY_PATTERNS = {
    # The word "gate" is NOT required: passengers write "I am at B12" just as often.
      ## Candidates are validated against the real gate names in kb_store, so a stray
      ## match cannot resolve to a gate that does not exist.
    "gate_number":     r"\b([a-z]\s?\d{1,2})\b",
    "terminal_number": r"\bterminal\s+(\d)\b",
    "time_reference":  r"\b(\d{1,2}[:h]\d{2}\s?(?:am|pm)?|\d{1,2}\s?(?:am|pm))\b",
    "flight_number":   r"\b([a-z]{2}\s?\d{3,4})\b",
}


# Run the shared text preprocessing pipeline on typed or transcribed input
def preprocess_text(text: str) -> Dict[str, Any]:
    # Reject empty or non-text input
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The text must be a non-empty string.")
    # Keep the original for the audit trace
    original = text.strip()
    # Lowercase, collapse whitespace, then convert spoken numbers to digits
    normalised = normalise_spoken_numbers(re.sub(r"\s+", " ", original.lower()))
    # Expand each known airport abbreviation
    for pattern, replacement in AIRPORT_ABBREVIATIONS.items():
        normalised = re.sub(pattern, replacement, normalised)
    # Extract the structured entities present in the query
    entities = {
        name: [match.strip().upper() for match in re.findall(pattern, normalised)]
        for name, pattern in ENTITY_PATTERNS.items()
        if re.findall(pattern, normalised)
    }
    # Return every intermediate result so the pipeline stays inspectable
    return {"original_text": original, "normalised_text": normalised, "entities": entities}
