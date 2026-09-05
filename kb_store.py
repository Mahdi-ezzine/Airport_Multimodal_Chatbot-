# Knowledge base loading and FAISS index construction for the deployed application
import json
import re
from typing import Any, Dict, List

import faiss
import numpy as np

from airport_config import CONFIG


# Load the structured knowledge base from its JSON file
def load_knowledge_base(path: str = CONFIG.kb_path) -> List[Dict[str, Any]]:
    # Open the file and parse its JSON content
    with open(path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    # Fail clearly if the knowledge base is empty
    if not records:
        raise ValueError("The knowledge base file contains no records.")
    # Return the list of records
    return records


# Define the retriever that owns both embedding spaces over the knowledge base
class AirportRetriever:
    # Build the indexes from the records and the two supplied encoder functions
    def __init__(self, records: List[Dict[str, Any]], clip_encoder, semantic_encoder):
        # Store the records in a fixed order shared by both indexes
        self.records = records
        # Extract both retrieval strings in that same shared order
        self.search_texts = [record["search_text"] for record in records]
        self.clip_texts = [record["clip_text"] for record in records]

        # Build the CLIP index over CATEGORIES rather than over records
          ## Several records share a category and differ only by a gate number or terminal,
          ## which no photograph can reveal. Indexing records would make the top matches tie
          ## on every image query. Categories are what an image can actually identify.
        self.categories = sorted({record["category"] for record in records})
        self.category_of_record = [self.categories.index(r["category"]) for r in records]
        self.category_matrix = self._build_category_matrix(self.categories, clip_encoder)
        # Each record inherits its category's vector, keeping the record ordering aligned
        self.clip_matrix = self.category_matrix[self.category_of_record]
        # Embed every record into sentence-transformer space for text questions
        self.semantic_matrix = semantic_encoder(self.search_texts)

        # Build the CLIP-space index using inner product over normalised vectors
        self.clip_index = faiss.IndexFlatIP(self.clip_matrix.shape[1])
        self.clip_index.add(self.clip_matrix)

        # Build the semantic-space index the same way
        self.semantic_index = faiss.IndexFlatIP(self.semantic_matrix.shape[1])
        self.semantic_index.add(self.semantic_matrix)

        # Verify both matrices contain unit-length vectors before anything relies on them
          ## Inner-product search equals cosine similarity only for unit vectors, and the
          ## confidence thresholds assume scores are bounded to [-1, 1]
        for space_name, matrix in (("clip", self.clip_matrix), ("semantic", self.semantic_matrix)):
            norms = np.linalg.norm(matrix, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                raise ValueError(
                    f"The {space_name} embedding matrix is not unit-normalised "
                    f"(norms {norms.min():.4f} to {norms.max():.4f})."
                )

    # Build one averaged CLIP embedding per signage category
    @staticmethod
    def _build_category_matrix(categories, clip_encoder) -> np.ndarray:
        # Phrase each category several different ways and average the embeddings
          ## Prompt ensembling: one caption is one arbitrary point in CLIP space, so the mean
          ## of several phrasings is a more representative centroid of the same concept
        templates = [
            "a photo of an airport sign for {category}",
            "an airport wayfinding sign pointing to {category}",
            "a directional sign inside an airport terminal for {category}",
            "airport signage indicating {category}",
            "a photograph of the {category} sign in an airport",
        ]
        vectors = []
        for category in categories:
            readable = category.replace("_", " ")
            prompts = [t.format(category=readable).lower() for t in templates]
            # Average the encoded templates, then renormalise back to unit length
              ## The mean of unit vectors is shorter than one, and a non-unit index would
              ## break the equivalence between inner product and cosine similarity
            mean_vector = clip_encoder(prompts).mean(axis=0)
            vectors.append(mean_vector / np.linalg.norm(mean_vector))
        return np.stack(vectors).astype("float32")

    # Score an image against the categories rather than the records
    def category_scores(self, image_vector: np.ndarray) -> np.ndarray:
        return (self.category_matrix @ image_vector.reshape(-1)).astype("float32")

    # Parse a gate identifier out of a record name or a query fragment
    @staticmethod
    def parse_gate(text):
        match = re.search(r"\b([A-Za-z])\s?(\d{1,2})\b", text)
        return (match.group(1).upper(), int(match.group(2))) if match else None

    # Apply an additive bonus to the records the named entities resolve to
    def apply_entity_boost(self, scores: np.ndarray, entities: Dict[str, Any]):
        # Reject anything other than the entity dictionary produced by preprocess_text
        if not isinstance(entities, dict):
            raise TypeError(
                f"apply_entity_boost expects the entity dictionary, not {type(entities).__name__}."
            )

        # Work on a copy so the raw similarity vector stays available for the confidence
          ## The boost affects RANKING only, never the reported confidence
        boosted = scores.copy()
        applied = []

        # Build the lookup of gate identifiers that actually exist, once per call
        gate_lookup = {}
        for index, record in enumerate(self.records):
            if record["category"] == "gate":
                parsed = self.parse_gate(record["name"])
                if parsed:
                    gate_lookup.setdefault(parsed, []).append(index)

        # A named gate is unambiguous, so it should decide the answer
          ## Comparing (letter, number) pairs means A5, A05 and "a five" all resolve alike,
          ## and a candidate matching no real gate is dropped rather than inventing one
        for candidate in entities.get("gate_number", []):
            parsed = self.parse_gate(candidate)
            for index in gate_lookup.get(parsed, []):
                boosted[index] += 1.00
                applied.append(f"{self.records[index]['record_id']} matches the gate you named")

        # A named terminal narrows the field without deciding it
        for terminal_number in entities.get("terminal_number", []):
            for index, record in enumerate(self.records):
                if terminal_number in record["terminal"]:
                    boosted[index] += 0.15
                    applied.append(f"{record['record_id']} is in the terminal you named")

        return boosted, applied

    # Return the similarity of a query vector against every record, in record order
    def similarity_vector(self, query_vector: np.ndarray, *, space: str) -> np.ndarray:
        # Select the matrix matching the requested embedding space
        if space == "clip":
            matrix = self.clip_matrix
        elif space == "semantic":
            matrix = self.semantic_matrix
        else:
            raise ValueError(f"Unknown embedding space: {space}")
        # Compute the dot product of the query against every record vector
        return (matrix @ query_vector.reshape(-1)).astype("float32")

    # Rank the records given a similarity vector already computed
    def rank_from_scores(self, scores: np.ndarray, *, top_k: int = CONFIG.top_k) -> List[Dict[str, Any]]:
        # Find the indices of the highest-scoring records
        best_indices = np.argsort(-scores)[:top_k]
        # Assemble the ranked results, attaching the score to each record
        return [
            {**self.records[int(index)], "score": float(scores[int(index)])}
            for index in best_indices
        ]
