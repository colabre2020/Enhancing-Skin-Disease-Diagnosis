from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
from PIL import Image, ImageStat
import math


DISEASE_CLASSES = [
    "melanoma",
    "melanocytic_nevus",
    "basal_cell_carcinoma",
    "actinic_keratosis",
    "benign_keratosis",
    "dermatofibroma",
    "vascular_lesion",
]


def _softmax(logits: Dict[str, float]) -> Dict[str, float]:
    max_logit = max(logits.values())
    exps = {key: math.exp(value - max_logit) for key, value in logits.items()}
    total = sum(exps.values())
    return {key: value / total for key, value in exps.items()}


def _bounded(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


@dataclass
class LightweightDiagnosis:
    predictions: Dict[str, float]
    confidence_scores: Dict[str, float]
    visual_concepts: Dict[str, float]
    clinical_concepts: Dict[str, float]
    explanation: str


class LightweightDermEngine:
    """Lightweight deterministic inference for serverless environments."""

    disease_classes = DISEASE_CLASSES

    def diagnose(self, image: Image.Image, clinical_history: str, patient_metadata: Dict) -> LightweightDiagnosis:
        rgb = image.convert("RGB")
        stats = ImageStat.Stat(rgb)
        means = stats.mean
        stds = stats.stddev

        mean_r, mean_g, mean_b = means
        std_r, std_g, std_b = stds

        brightness = _bounded((mean_r + mean_g + mean_b) / (3 * 255.0))
        chroma = _bounded(abs(mean_r - mean_b) / 255.0)
        heterogeneity = _bounded((std_r + std_g + std_b) / (3 * 128.0))
        border_irregularity = _bounded(heterogeneity * 0.9 + chroma * 0.2)
        asymmetry = _bounded(0.45 * heterogeneity + 0.25 * chroma + 0.2 * (1 - brightness))
        color_variation = _bounded(0.6 * chroma + 0.4 * heterogeneity)
        diameter = _bounded(0.35 + 0.4 * heterogeneity)

        age = int(patient_metadata.get("age", 50) or 50)
        age_factor = _bounded(age / 90.0)

        history = f"{clinical_history} {patient_metadata.get('symptoms', '')}".lower()

        keyword_signal = {
            "melanoma": any(word in history for word in ["bleeding", "change", "growing", "irregular", "dark"]),
            "basal_cell_carcinoma": any(word in history for word in ["pearly", "ulcer", "shiny", "non-healing"]),
            "actinic_keratosis": any(word in history for word in ["sun", "rough", "scaly", "crust"]),
            "vascular_lesion": any(word in history for word in ["red", "vascular", "purple"]),
            "benign_keratosis": any(word in history for word in ["waxy", "stuck", "keratosis"]),
            "dermatofibroma": any(word in history for word in ["firm", "nodule", "dimple"]),
            "melanocytic_nevus": any(word in history for word in ["mole", "stable", "long-standing"]),
        }

        logits = {
            "melanoma": 1.0 + 1.4 * asymmetry + 1.2 * color_variation + 0.8 * age_factor,
            "melanocytic_nevus": 1.1 + 0.8 * brightness + 0.5 * (1 - heterogeneity),
            "basal_cell_carcinoma": 0.9 + 0.9 * border_irregularity + 0.9 * age_factor,
            "actinic_keratosis": 0.85 + 0.9 * age_factor + 0.7 * (1 - brightness),
            "benign_keratosis": 0.8 + 0.6 * brightness + 0.5 * heterogeneity,
            "dermatofibroma": 0.75 + 0.6 * (1 - color_variation) + 0.4 * heterogeneity,
            "vascular_lesion": 0.75 + 1.1 * _bounded((mean_r - mean_g) / 255.0) + 0.4 * chroma,
        }

        for disease, has_signal in keyword_signal.items():
            if has_signal:
                logits[disease] += 0.55

        predictions = _softmax(logits)
        top_class = max(predictions, key=predictions.get)
        top_prob = predictions[top_class]

        uncertainty = _bounded(1.0 - top_prob)
        overall_confidence = _bounded(0.55 + 0.45 * top_prob)

        visual_concepts = {
            "asymmetry": asymmetry,
            "border_irregularity": border_irregularity,
            "color_variation": color_variation,
            "diameter": diameter,
        }

        clinical_concepts = {
            "patient_age": age_factor,
            "lesion_location": 0.5,
            "skin_type": 0.5,
        }

        explanation = (
            f"Lightweight serverless assessment suggests {top_class.replace('_', ' ')} "
            f"with confidence {top_prob:.1%}. This Vercel mode uses image-statistics and "
            f"clinical-text heuristics (not deep-learning inference), so treat as decision support only."
        )

        return LightweightDiagnosis(
            predictions=predictions,
            confidence_scores={
                "overall_confidence": overall_confidence,
                "uncertainty": uncertainty,
            },
            visual_concepts=visual_concepts,
            clinical_concepts=clinical_concepts,
            explanation=explanation,
        )
