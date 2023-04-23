import subprocess
import os
from spacy.cli.train import train


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
    def startTraining(isResumed: bool):
        directory = "../output"
        if not os.path.exists(directory):
            os.mkdir(directory)

        configPath: str = "../config/config.cfg"
        if isResumed:
            configPath = "../config/resume_config.cfg"
        subprocess.run(
            [
                "python",
                "-m",
                "spacy",
                "train",
                configPath,
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
