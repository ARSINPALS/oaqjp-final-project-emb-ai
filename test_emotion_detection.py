"""Unit tests for the EmotionDetection package."""

import unittest
from unittest.mock import Mock, patch

from EmotionDetection import emotion_detection


class TestEmotionDetection(unittest.TestCase):
    """Test cases for emotion_detector function."""

    def mock_response(self, emotion_scores, status_code=200):
        """Create a mocked HTTP response."""
        response = Mock()
        response.status_code = status_code
        response.json.return_value = {
            "emotionPredictions": [{"emotion": emotion_scores}]
        }
        return response

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_emotion_detector(self, mock_post):
        """Verify dominant emotion for each target emotion."""
        cases = (
            ("I am glad this happened", "joy",
             {"anger": 0.1, "disgust": 0.1, "fear": 0.1, "joy": 0.9, "sadness": 0.1}),
            ("I am really mad about this", "anger",
             {"anger": 0.9, "disgust": 0.1, "fear": 0.1, "joy": 0.1, "sadness": 0.1}),
            ("I feel disgusted just hearing about this", "disgust",
             {"anger": 0.1, "disgust": 0.9, "fear": 0.1, "joy": 0.1, "sadness": 0.1}),
            ("I am so sad about this", "sadness",
             {"anger": 0.1, "disgust": 0.1, "fear": 0.1, "joy": 0.1, "sadness": 0.9}),
            ("I am really afraid that this will happen", "fear",
             {"anger": 0.1, "disgust": 0.1, "fear": 0.9, "joy": 0.1, "sadness": 0.1}),
        )

        for text, dominant, scores in cases:
            with self.subTest(dominant=dominant):
                mock_post.return_value = self.mock_response(scores)
                result = emotion_detection.emotion_detector(text)
                self.assertEqual(result["dominant_emotion"], dominant)

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_empty_text_returns_null_scores(self, mock_post):
        """Verify invalid text does not call the external service."""
        result = emotion_detection.emotion_detector("")
        self.assertIsNone(result["dominant_emotion"])
        mock_post.assert_not_called()

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_bad_request_returns_null_scores(self, mock_post):
        """Verify HTTP 400 responses return null emotion scores."""
        mock_post.return_value = self.mock_response({}, status_code=400)
        result = emotion_detection.emotion_detector("test")
        self.assertIsNone(result["dominant_emotion"])


if __name__ == "__main__":
    unittest.main()