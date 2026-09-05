# Routing, weighted similarity fusion and response formatting for the deployed application
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from airport_config import CONFIG, LIVE_DATA_DISCLAIMER, UNCERTAIN_RESPONSE


# Define the structure of a single chatbot response
@dataclass
class AirportResponse:
    # The passenger-facing answer text
    answer: str
    # Whether the system was confident enough to answer at all
    answered: bool
    # The confidence score behind the answer
    confidence: float
    # The full audit trail of how the answer was produced
    trace: Dict[str, Any] = field(default_factory=dict)
    # The knowledge base records considered, best first
    sources: List[Dict[str, Any]] = field(default_factory=list)


# Rescale a similarity vector to the range 0 to 1
def normalise_scores(scores: np.ndarray) -> np.ndarray:
    # Find the smallest and largest score present
    minimum, maximum = float(scores.min()), float(scores.max())
    # Guard against a flat vector, which would cause a division by zero
    if maximum - minimum < 1e-8:
        return np.zeros_like(scores)
    # Rescale linearly so the best record scores 1 and the worst scores 0
      ## Used ONLY to place two modalities on a comparable scale before ranking.
      ## Never reported as a confidence: the top record always rescales to exactly 1.0.
    return ((scores - minimum) / (maximum - minimum)).astype("float32")


# Decide which encoders to run based on which inputs are present
def route_by_modality(*, image=None, text: Optional[str] = None, audio_path: Optional[str] = None) -> str:
    # Record which modalities are actually present
    has_image = image is not None
    # Voice input is converted to text, so it joins the text branch
    text_present = bool(text and text.strip()) or bool(audio_path and audio_path.strip())
    # Select the route based on the combination of available inputs
    if has_image and text_present:
        return "fusion"
    if has_image:
        return "image_only"
    if text_present:
        return "text_only"
    # Refuse explicitly when no input was supplied at all
    return "no_input"


# Turn one knowledge base record into a passenger-facing answer
def format_airport_answer(record: Dict[str, Any]) -> str:
    # Assemble the answer from the fields the brief requires in the fused response
    lines = [
        f"**{record['name']}** — {record['terminal']}, {record['floor_zone']}",
        "",
        record["description"],
        "",
        f"**Directions:** {record['direction_text']}",
        f"**Opening hours:** {record['opening_hours']}",
        f"**Accessibility:** {record['accessibility']}",
    ]
    # Add nearby services as a recommendation when the record lists any
    if record.get("related_facilities"):
        lines.append(f"**Nearby:** {', '.join(record['related_facilities'])}")
    # Always attach the live-data reminder and the source citation
    lines.append("")
    lines.append(f"_{LIVE_DATA_DISCLAIMER}_")
    lines.append(f"_Source: {record['record_id']}_")
    # Join the lines into one answer string
    return "\n".join(lines)


# Turn any combination of inputs into one grounded response
def answer_passenger_query(
    retriever, encoders, *, image=None, text: Optional[str] = None, audio_path: Optional[str] = None
) -> AirportResponse:
    # Unpack the encoder functions supplied by the caller
    encode_images, _, encode_text_semantic, transcribe_audio, preprocess_text = encoders
    # Start the audit trace that will record every decision taken
    trace: Dict[str, Any] = {}

    # 1. Decide which pathway to run
    route = route_by_modality(image=image, text=text, audio_path=audio_path)
    trace["route"] = route

    # Refuse immediately if the passenger supplied nothing at all
    if route == "no_input":
        return AirportResponse(
            answer="Please type a question, upload a voice recording, or upload a photo of a sign.",
            answered=False, confidence=0.0, trace=trace,
        )

    # 2. Convert any voice input into text using Whisper
    query_text = text.strip() if text and text.strip() else None
    if audio_path and audio_path.strip():
        transcription = transcribe_audio(audio_path)
        trace["transcription"] = transcription
        query_text = query_text or transcription["transcript"]

    # 3. Run the shared text preprocessing pipeline
    text_scores = None
    if query_text:
        preprocessed = preprocess_text(query_text)
        trace["normalised_text"] = preprocessed["normalised_text"]
        trace["entities"] = preprocessed["entities"]
        text_vector = encode_text_semantic([preprocessed["normalised_text"]])[0]
        text_scores = retriever.similarity_vector(text_vector, space="semantic")
        trace["text_best_score"] = round(float(text_scores.max()), 4)

        # Boost records matching any terminal or gate the passenger named
          ## The boost affects RANKING only. The confidence reported later is still read from
          ## the unboosted raw similarities, so an artificial bonus can never inflate it.
        text_ranking_scores, boosts = retriever.apply_entity_boost(text_scores, preprocessed["entities"])
        if boosts:
            trace["entity_boosts"] = boosts

    # 4. Encode any image and score it against the signage CATEGORIES
    image_scores = None
    category_margin = None
    if image is not None:
        image_vector = encode_images([image])[0]
        category_similarity = retriever.category_scores(image_vector)
        # Measure how clearly the winning category beat the runner-up
        ordered_categories = np.sort(category_similarity)[::-1]
        category_margin = float(ordered_categories[0] - ordered_categories[1])
        # Spread each category score across the records belonging to it
        image_scores = category_similarity[retriever.category_of_record]
        trace["vision"] = {
            "predicted_category": retriever.categories[int(np.argmax(category_similarity))],
            "category_similarity": round(float(category_similarity.max()), 4),
            "category_margin": round(category_margin, 4),
        }

    # 5. Combine the available evidence for ranking only
    if image_scores is not None and text_scores is not None:
        ranking = (CONFIG.image_weight * normalise_scores(image_scores)
                   + CONFIG.text_weight * normalise_scores(text_ranking_scores))
        trace["fusion_strategy"] = "weighted similarity over aligned record order"
        threshold = (CONFIG.image_weight * CONFIG.image_confidence_threshold
                     + CONFIG.text_weight * CONFIG.text_confidence_threshold)
    elif image_scores is not None:
        ranking = image_scores
        trace["fusion_strategy"] = "image evidence only"
        threshold = CONFIG.image_confidence_threshold
    else:
        ranking = text_ranking_scores
        trace["fusion_strategy"] = "text evidence only"
        threshold = CONFIG.text_confidence_threshold

    # 6. Rank the knowledge base records and locate the winner
    sources = retriever.rank_from_scores(ranking, top_k=CONFIG.top_k)
    winner_index = int(np.argmax(ranking))

    # 7. Compute the confidence from the RAW similarities of the winning record
      ## Raw cosine similarity is comparable across queries, which is what makes a
      ## fixed threshold meaningful. The min-max ranking score never could be.
    raw_parts = {}
    if image_scores is not None:
        raw_parts["image"] = float(image_scores[winner_index])
    if text_scores is not None:
        raw_parts["text"] = float(text_scores[winner_index])
    if len(raw_parts) == 2:
        confidence = CONFIG.image_weight * raw_parts["image"] + CONFIG.text_weight * raw_parts["text"]
    else:
        confidence = float(next(iter(raw_parts.values())))

    # Measure how clearly the winner beat the runner-up
      ## On an image-only query the margin is measured over CATEGORIES: several records can
      ## legitimately share the winning category, and a tie between them is not uncertainty
      ## about what the sign says, only about which terminal the passenger is standing in
    if route == "image_only" and category_margin is not None:
        margin = category_margin
    else:
        ordered = np.sort(ranking)[::-1]
        margin = float(ordered[0] - ordered[1]) if ordered.size > 1 else float(ordered[0])

    # Note when several records share the winning category
    winning_category = sources[0]["category"] if sources else None
    tied = [r for r in retriever.records if r["category"] == winning_category]
    trace["records_in_winning_category"] = len(tied)

    # Record every number that drove the decision
    trace["raw_similarity"] = {k: round(v, 4) for k, v in raw_parts.items()}
    trace["confidence"] = round(confidence, 4)
    trace["threshold_applied"] = round(threshold, 4)
    trace["margin_top1_top2"] = round(margin, 4)

    # 8. Refuse to answer when the raw confidence is below the modality threshold
    if confidence < threshold:
        trace["answered"] = False
        trace["refusal_reason"] = "confidence below threshold"
        return AirportResponse(UNCERTAIN_RESPONSE, False, confidence, trace, sources)

    # 9. Build the grounded passenger-facing answer
    trace["answered"] = True
    trace["source_record_id"] = sources[0]["record_id"]

    # Ask which terminal the passenger is in when the category holds several records
      ## An image identifies the kind of place but not which one, so the system says so
      ## rather than presenting an arbitrary choice with false confidence
    extra = ""
    if route == "image_only" and len(tied) > 1:
        # [BUG FIX] Compare by record_id, not by object identity: rank_from_scores builds
          ## NEW dicts, so "r is not sources[0]" was always true and the terminal already
          ## being shown was listed among the alternatives to choose from
        others = ", ".join(r["terminal"] for r in tied if r["record_id"] != sources[0]["record_id"])
        extra = (f"\n\n**This sign appears in more than one place.** I have shown the one in "
                 f"{sources[0]['terminal']}. Tell me your terminal ({others}) for the right one.")
    return AirportResponse(format_airport_answer(sources[0]) + extra, True, confidence, trace, sources)
