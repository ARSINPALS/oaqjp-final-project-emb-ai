"""Unit tests for the EmotionDetection package."""

import unittest
from EmotionDetection import emotion_detection


class TestEmotionDetection(unittest.TestCase):
    """Test cases for emotion_detector function."""

    def test_emotion_detector(self):
        """Verify dominant emotion for each target emotion."""
        res_joy = emotion_detection.emotion_detector(
            "I am glad this happened"
        )
        self.assertEqual(res_joy["dominant_emotion"], "joy")

        res_anger = emotion_detection.emotion_detector(
            "I am really mad about this"
        )
        self.assertEqual(res_anger["dominant_emotion"], "anger")

        res_disgust = emotion_detection.emotion_detector(
            "I feel disgusted just hearing about this"
        )
        self.assertEqual(res_disgust["dominant_emotion"], "disgust")

        res_sadness = emotion_detection.emotion_detector(
            "I am so sad about this"
        )
        self.assertEqual(res_sadness["dominant_emotion"], "sadness")

        res_fear = emotion_detection.emotion_detector(
            "I am really afraid that this will happen"
        )
        self.assertEqual(res_fear["dominant_emotion"], "fear")


if __name__ == "__main__":
    unittest.main()