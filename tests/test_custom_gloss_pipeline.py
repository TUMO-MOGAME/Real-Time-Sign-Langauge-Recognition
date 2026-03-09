import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.gloss.backbone import get_model
from scripts.train_custom_gloss import build_label_map, discover_sequences, flatten_examples


class CustomGlossPipelineTests(unittest.TestCase):
    def test_get_model_supports_custom_class_count(self):
        model = get_model(max_len=None, num_classes=25)
        self.assertEqual(model.output_shape[-1], 25)

    def test_discover_sequences_ignores_extra_files_and_requires_exact_frames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "MP_Data"
            sequence_dir = root / "Week_1_Greetings" / "hello" / "sample_0"
            sequence_dir.mkdir(parents=True)
            for index in range(30):
                np.save(sequence_dir / f"{index}.npy", np.zeros(1662, dtype=np.float32))
            np.save(sequence_dir / "17(1).npy", np.zeros(1662, dtype=np.float32))

            incomplete_dir = root / "Week_1_Greetings" / "welcome" / "sample_1"
            incomplete_dir.mkdir(parents=True)
            for index in range(29):
                np.save(incomplete_dir / f"{index}.npy", np.zeros(1662, dtype=np.float32))

            sign_to_sequences, skipped = discover_sequences(root)
            self.assertEqual(list(sign_to_sequences.keys()), ["hello"])
            self.assertEqual(len(sign_to_sequences["hello"]), 1)
            self.assertEqual(len(skipped), 1)

    def test_label_map_and_flatten_examples_are_deterministic(self):
        sign_to_sequences = {
            "welcome": ["seq_b"],
            "alive": ["seq_a1", "seq_a2"],
        }
        label_map = build_label_map(sign_to_sequences)
        paths, labels = flatten_examples(sign_to_sequences, label_map)

        self.assertEqual(label_map, {"alive": 0, "welcome": 1})
        self.assertEqual(paths, ["seq_a1", "seq_a2", "seq_b"])
        self.assertEqual(labels.tolist(), [0, 0, 1])


if __name__ == "__main__":
    unittest.main()