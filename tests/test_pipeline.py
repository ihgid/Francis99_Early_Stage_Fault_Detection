import unittest
import numpy as np
from src.preprocess import preprocess_single_reading
from src.llm_expert import get_expert_interpretation

class DummyScaler:
    def transform(self, X):
        return X / 100.0

class TestPipeline(unittest.TestCase):
    def test_preprocess_shape(self):
        scaler = DummyScaler()
        sensor_values = [0.12, 0.45, -0.05, 0.88, 0.33]
        processed = preprocess_single_reading(sensor_values, scaler, sequence_length=30)
        self.assertEqual(processed.shape, (1, 30, 5))

    def test_fallback_expert(self):
        res = get_expert_interpretation(1, api_key=None)
        self.assertIn("stage", res)
        self.assertIn("15 mm", res["crack_length"])

if __name__ == '__main__':
    unittest.main()
