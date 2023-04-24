import spacy
from spacy.tokens import DocBin
import subprocess


class Inference:
    def __init__(self):
        self.model_path = "../output/model-best"
        self.model = spacy.load(self.model_path)
        self.default_test_dataset_path = "../data/test_data.spacy"
        self.default_test_dataset = DocBin().from_disk(self.default_test_dataset_path)

    def evaluateTestDataset(self, test_file_location):
        if not test_file_location:
            test_file_location = self.default_test_dataset_path

        subprocess.run(
            [
                "python",
                "-m",
                "spacy",
                "evaluate",
                self.model_path,
                test_file_location,
                "--output",
                "../output/evaluate.json",
                "--gpu-id",
                "0",
                "--code",
                "./custom_logger.py",
            ]
        )
