import sys
from typing import IO, Tuple, Callable, Dict, Any, Optional
import spacy
from spacy import Language
from pathlib import Path
from spacy.training.loggers import console_logger
import tensorflow as tf


@spacy.registry.loggers("custom_logger.v1")
def custom_logger(log_path):
    console = console_logger(progress_bar=True)
    train_writer = tf.summary.create_file_writer('../log/train')

    def setup_logger(
            nlp: Language,
            stdout: IO = sys.stdout,
            stderr: IO = sys.stderr,
    ) -> Tuple[Callable, Callable]:

        stdout.write(f"Logging to {log_path}\n")
        console_log_step, console_finalize = console(nlp, stdout, stderr)
        log_file = Path(log_path).open("w", encoding="utf8")
        log_file.write("step\t")
        log_file.write("score\t")
        for pipe in nlp.pipe_names:
            log_file.write(f"loss_{pipe}\t")
        log_file.write("\n")
        log_file.flush()

        def log_step(info: Optional[Dict[str, Any]]):
            console_log_step(info)
            if info:
                log_file.write(f"{info['step']}\t")
                log_file.write(f"{info['score']}\t")
                for pipe in nlp.pipe_names:
                    # Write for tensorboard
                    with train_writer.as_default():
                        tf.summary.scalar(f'loss_{pipe}', info['losses'][pipe], step=info['step'])
                    log_file.write(f"{info['losses'][pipe]}\t")
                log_file.write("\n")
                log_file.flush()
                # Write for tensorboard
                with train_writer.as_default():
                    tf.summary.scalar('Score', info['score'], step=info['step'])
                    train_writer.flush()

        def finalize():
            console_finalize()
            log_file.close()
            train_writer.close()

        return log_step, finalize

    return setup_logger
