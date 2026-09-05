# Streamlit user interface for the airport passenger assistance chatbot
import os
import tempfile

import streamlit as st
from PIL import Image

import html
import re

from airport_config import CONFIG, PRIVACY_NOTICE
from fusion import answer_passenger_query
from kb_store import AirportRetriever, load_knowledge_base
from perception import build_encoders, load_models, preprocess_text

# Configure the page before any other Streamlit call
st.set_page_config(page_title="Airport Assistant", page_icon="✈️", layout="wide")

# Enlarge the typography for readability
  ## A passenger reads this on a phone, often in a hurry and often in poor light.
  ## Streamlit's defaults are sized for dashboards, not for wayfinding instructions.
st.markdown('''
<style>
  html, body, [class*="css"], .stMarkdown, .stMarkdown p, .stMarkdown li { font-size: 1.15rem; }
  .stMarkdown h1 { font-size: 2.4rem; }
  .stMarkdown h2 { font-size: 1.8rem; }
  .stMarkdown h3 { font-size: 1.45rem; }
  .answer-card {
      font-size: 1.25rem; line-height: 1.75; padding: 1.4rem 1.6rem;
      border-radius: 12px; border-left: 6px solid #0052A3; background: rgba(0, 82, 163, 0.07);
  }
  .answer-card strong { font-size: 1.3rem; }
  div.stButton > button { font-size: 1.1rem; padding: 0.6rem 1rem; }
  /* Streamlit's default primary red reads as a warning on a passenger-assistance screen */
  div.stButton > button[kind="primary"] {
      background-color: #0052A3; border-color: #0052A3; color: #ffffff; font-weight: 600;
  }
  div.stButton > button[kind="primary"]:hover { background-color: #004080; border-color: #004080; }
  .stTextInput input { font-size: 1.2rem; padding: 0.7rem; }
  section[data-testid="stFileUploaderDropzone"] { padding: 1.1rem; }
</style>
''', unsafe_allow_html=True)


# Convert the answer's markdown into HTML for display inside the styled card
def markdown_to_html(text: str) -> str:
    # [BUG FIX] Streamlit does NOT convert markdown inside a raw HTML block passed with
      ## unsafe_allow_html, so "**Gate B12**" was rendering with its asterisks visible and
      ## every line break was collapsing into one run-on paragraph. The answer uses only
      ## three constructs, all produced by format_airport_answer, so converting them here
      ## is exact rather than a general-purpose markdown parser.
    # Escape any HTML the knowledge base might contain, before adding our own
    safe = html.escape(text)
    # Convert bold, then italic, then line breaks
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    safe = re.sub(r"_(.+?)_", r"<em>\1</em>", safe)
    return safe.replace("\n\n", "<br><br>").replace("\n", "<br>")


# Load the models once and keep them in memory across reruns
@st.cache_resource(show_spinner="Loading multimodal models, this takes a moment on first run...")
def get_system():
    # Load every frozen model
    clip_processor, clip_model, text_model, speech_model = load_models()
    # Build the encoding and transcription functions
    encode_images, encode_text_clip, encode_text_semantic, transcribe_audio = build_encoders(
        clip_processor, clip_model, text_model, speech_model
    )
    # Load the knowledge base and build both FAISS indexes
    records = load_knowledge_base()
    retriever = AirportRetriever(records, encode_text_clip, encode_text_semantic)
    # Bundle the encoders in the order the fusion module expects
    encoders = (encode_images, encode_text_clip, encode_text_semantic, transcribe_audio, preprocess_text)
    # Return the assembled system
    return retriever, encoders


# Initialise the conversation history on first run
if "history" not in st.session_state:
    st.session_state.history = []

# Draw the page header
st.title("✈️ Airport Passenger Assistance")
st.caption("Ask by text, upload a voice recording, or photograph a sign. "
           "Powered by CLIP, Whisper and FAISS over a 36-record airport knowledge base.")

# Load the system before drawing anything that depends on it
retriever, encoders = get_system()

# Put the reference information in the sidebar, out of the main flow
with st.sidebar:
    st.subheader("About this assistant")
    st.write("This is a proof-of-concept multimodal chatbot. It answers only from a fixed "
             "airport knowledge base and refuses when it is not confident enough.")

    # Show the thresholds so the confidence figures shown below are interpretable
    st.subheader("Confidence thresholds")
    st.write(f"Text queries: **{CONFIG.text_confidence_threshold:.2f}**")
    st.write(f"Image queries: **{CONFIG.image_confidence_threshold:.2f}**")
    st.caption("Raw cosine similarity. Two thresholds are used because the text and image "
               "encoders produce different score ranges.")

    # Show what the assistant does and does not cover
    st.subheader("Coverage")
    categories = sorted({record["category"].replace("_", " ") for record in retriever.records})
    st.write(", ".join(categories))

    st.divider()
    st.caption(PRIVACY_NOTICE)

# Initialise the conversation history on first run
if "history" not in st.session_state:
    st.session_state.history = []
# Seed the question box once, so the example buttons can write into it safely
  ## [BUG FIX] Passing value= to a widget that also has key= conflicts with Streamlit's
  ## own session state, and the example text silently failed to appear. Writing to
  ## st.session_state[key] BEFORE the widget is created is the supported pattern.
if "text_query" not in st.session_state:
    st.session_state.text_query = ""

# Split the page into an input column and a response column
input_column, response_column = st.columns([1, 1.4])

with input_column:
    st.subheader("Your question")

    # Offer example questions, since a blank box gives the passenger no idea what is supported
    st.caption("Try an example:")
    example_columns = st.columns(2)
    EXAMPLES = [
        "Where is gate B12?",
        "How do I get to baggage claim?",
        "I feel unwell, where can I get help?",
        "I need wheelchair assistance",
    ]
    for position, example in enumerate(EXAMPLES):
        if example_columns[position % 2].button(example, key=f"example_{position}",
                                                use_container_width=True):
            # Write straight into the widget's own state, then rerun so it picks it up
            st.session_state.text_query = example
            st.rerun()

    # Text input for typed passenger questions
    text_query = st.text_input("Type your question",
                               placeholder="Where is gate B12?", key="text_query")

    # Image upload for photographs of airport signage
    uploaded_image = st.file_uploader("Upload a photo of a sign", type=["png", "jpg", "jpeg"],
                                      key="image_upload")
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded sign", width=280)

    # Audio upload for recorded passenger questions
    uploaded_audio = st.file_uploader("Upload a voice recording", type=["mp3", "wav", "m4a"],
                                      key="audio_upload")
    if uploaded_audio is not None:
        st.audio(uploaded_audio)

    # The button that runs the pipeline
    submitted = st.button("Ask the assistant", type="primary", use_container_width=True)
    # The button that clears the session
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.text_query = ""
        st.rerun()

with response_column:
    st.subheader("Assistant response")

    # Prompt the passenger when nothing has been asked yet
    if not st.session_state.history and not submitted:
        st.info("Ask a question on the left, or upload a photo of an airport sign.", icon="👈")

    # Run the pipeline when the button is pressed
    if submitted:
        # Load the uploaded image into memory, never to disk
        image_object = Image.open(uploaded_image).convert("RGB") if uploaded_image else None

        # Persist the uploaded audio to a temporary file, since Whisper reads from disk
        audio_path = None
        if uploaded_audio is not None:
            suffix = os.path.splitext(uploaded_audio.name)[1] or ".mp3"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                handle.write(uploaded_audio.getbuffer())
                audio_path = handle.name

        # Run the full multimodal pipeline
        with st.spinner("Analysing your question..."):
            response = answer_passenger_query(
                retriever, encoders,
                image=image_object, text=text_query or None, audio_path=audio_path,
            )

        # Delete the temporary audio file immediately, as promised in the privacy notice
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

        # Store the exchange in the session history
        st.session_state.history.insert(0, response)

    # Render every exchange, most recent first
    for position, response in enumerate(st.session_state.history):
        # Translate the raw confidence into language a passenger can act on
          ## [DESIGN] A passenger has no way to interpret "0.16 versus a threshold of 0.22".
          ## The number is meaningful to an examiner and meaningless to a traveller, so the
          ## interface states what the system knows in words and keeps every figure in the
          ## technical panel below, where it remains available for the report and for audit.
        threshold = response.trace.get("threshold_applied", 0.0)
        headroom = response.confidence - threshold

        if response.answered:
            # Describe the strength of the match in plain words
            if headroom > 0.12:
                st.success("I am confident about this", icon="✅")
            elif headroom > 0.04:
                st.success("This is most likely what you are looking for", icon="✅")
            else:
                st.warning("This is my best match, but please double-check the signs nearby", icon="🔎")

            # Show the answer itself in a large, readable card
            st.markdown(f"<div class='answer-card'>{markdown_to_html(response.answer)}</div>",
                        unsafe_allow_html=True)
        else:
            # Refuse clearly, and offer the human handover path
            st.warning(response.answer, icon="⚠️")
            st.markdown("#### Would you like to speak to a person?")
            st.markdown("The **Central Information Desk** is on Level 1 of the main concourse, "
                        "opposite the departure boards. Staff are there from 05:00 to 23:00.")

        # Show what was heard, so a passenger can correct a misheard voice query
        if "transcription" in response.trace:
            st.caption(f"I heard: \"{response.trace['transcription']['transcript']}\"")

        # Say what the photograph was understood to show
        if "vision" in response.trace:
            seen = response.trace["vision"]["predicted_category"].replace("_", " ")
            st.caption(f"From your photo, this looks like a **{seen}** sign.")

        # Offer the alternatives, in passenger language rather than as scores
        if response.answered and response.sources and len(response.sources) > 1:
            with st.expander("Did you mean somewhere else?"):
                for source in response.sources[1:]:
                    st.markdown(f"**{source['name']}** — {source['terminal']}, {source['floor_zone']}")

        # Keep every technical figure available, but out of the passenger's way
        with st.expander("Technical details (for evaluation)"):
            detail_columns = st.columns(4)
            detail_columns[0].metric("Confidence", f"{response.confidence:.3f}")
            detail_columns[1].metric("Threshold", f"{threshold:.3f}")
            detail_columns[2].metric("Margin", f"{response.trace.get('margin_top1_top2', 0):.3f}")
            detail_columns[3].metric("Route", response.trace.get("route", "unknown"))
            if response.trace.get("entity_boosts"):
                st.caption("Entity boosts: " + "; ".join(response.trace["entity_boosts"][:4]))
            st.json(response.trace)

        st.divider()
