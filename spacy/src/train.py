import subprocess
import os


class Train:
    def __init__(self):
        pass

    @staticmethod
    def initializeConfig():
        subprocess.run(
            [
                "python",
                "-m",
                "spacy",
                "init",
                "fill-config",
                "../config/base_config.cfg",
                "../config/new_config.cfg",
            ]
        )

    @staticmethod
    def downloadPretrainedModel():
        subprocess.run(
            [
                "python",
                "-m",
                "spacy",
                "download",
                "en_core_web_lg",
            ]
        )

    @staticmethod
    def startTraining():
        directory = "../output"
        if not os.path.exists(directory):
            os.mkdir(directory)

        subprocess.run(
            [
                "python",
                "-m",
                "spacy",
                "train",
                "../config/config.cfg",
                "--output",
                "../output",
                "--paths.train",
                "../data/training_data.spacy",
                "--paths.dev",
                "../data/dev_data.spacy",
                "--gpu-id",
                "0",
                "--code",
                "./custom_logger.py",
            ]
        )
