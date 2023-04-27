"""
Create dataset object from json.
Split into train, validation and test set.
Add sentence and label
Add tokenization column
Convert to tf_dataset
"""

from datasets import load_dataset
from src.utility import Constants
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification


class TfDataset:
    tokenizer = AutoTokenizer.from_pretrained(Constants.MODEL_CHECKPOINT)

    def __init__(self, json_data_file):
        self.final_test_set = None
        self.final_validation_set = None
        self.final_train_set = None
        self.train_ds = load_dataset(
            "json", data_files=json_data_file, split=f"train[:70%]"
        )
        self.val_ds = load_dataset(
            "json", data_files=json_data_file, split=f"train[70%:90%]"
        )
        self.test_ds = load_dataset(
            "json", data_files=json_data_file, split="train[90%:]"
        )
        self.tokenized_train_ds = self.train_ds.map(
            TfDataset.tokenize_and_align_labels, batched=True
        )
        self.tokenized_val_ds = self.val_ds.map(
            TfDataset.tokenize_and_align_labels, batched=True
        )
        self.tokenized_test_ds = self.test_ds.map(
            TfDataset.tokenize_and_align_labels, batched=True
        )

    def forward(self, model):
        data_collator = DataCollatorForTokenClassification(
            TfDataset.tokenizer, return_tensors="np"
        )
        self.final_train_set = model.prepare_tf_dataset(
            self.tokenized_train_ds,
            shuffle=True,
            batch_size=Constants.BATCH_SIZE,
            collate_fn=data_collator,
        )

        self.final_validation_set = model.prepare_tf_dataset(
            self.tokenized_val_ds,
            shuffle=False,
            batch_size=Constants.BATCH_SIZE,
            collate_fn=data_collator,
        )

        self.final_test_set = model.prepare_tf_dataset(
            self.tokenized_test_ds,
            shuffle=False,
            batch_size=Constants.BATCH_SIZE,
            collate_fn=data_collator,
        )

    @staticmethod
    def tokenize_and_align_labels(examples):
        label_all_tokens = True
        task = Constants.TASK

        tokenized_inputs = TfDataset.tokenizer(examples["sentence"], truncation=True)

        labels = []
        for i, label in enumerate(examples[f"{task}_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    try:
                        label_ids.append(label[word_idx])
                    except IndexError:
                        label_ids.append(-100)
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    try:
                        label_ids.append(label[word_idx] if label_all_tokens else -100)
                    except IndexError:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
