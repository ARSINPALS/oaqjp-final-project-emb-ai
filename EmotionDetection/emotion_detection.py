"""Emotion detection module using Watson NLP."""

import json
import requests


def _empty_result():
    """Return a result used when emotion analysis cannot be performed."""
    return {
        "anger": None,
        "disgust": None,
        "fear": None,
        "joy": None,
        "sadness": None,
        "dominant_emotion": None,
    }


def emotion_detector(text_to_analyze):
    """Analyze emotions in the provided text and return formatted scores."""
    if not isinstance(text_to_analyze, str) or not text_to_analyze.strip():
        return _empty_result()

    url = (
        "https://sn-watson-emotion.labs.skills.network/"
        "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )
    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }
    payload = {"raw_document": {"text": text_to_analyze}}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
    except requests.RequestException:
        return _empty_result()

    if response.status_code == 400:
        return _empty_result()
    if response.status_code != 200:
        return _empty_result()

    try:
        try:
            formatted_response = response.json()
        except (AttributeError, ValueError, requests.RequestException):
            formatted_response = json.loads(response.text)
        emotions = formatted_response["emotionPredictions"][0]["emotion"]
    except (json.JSONDecodeError, ValueError, KeyError, IndexError, TypeError, AttributeError,
            requests.RequestException):
        return _empty_result()

    anger_score = emotions["anger"]
    disgust_score = emotions["disgust"]
    fear_score = emotions["fear"]
    joy_score = emotions["joy"]
    sadness_score = emotions["sadness"]

    scores = {
        "anger": anger_score,
        "disgust": disgust_score,
        "fear": fear_score,
        "joy": joy_score,
        "sadness": sadness_score,
    }
    dominant_emotion = max(scores, key=scores.get)

    return {
        "anger": anger_score,
        "disgust": disgust_score,
        "fear": fear_score,
        "joy": joy_score,
        "sadness": sadness_score,
        "dominant_emotion": dominant_emotion,
    }