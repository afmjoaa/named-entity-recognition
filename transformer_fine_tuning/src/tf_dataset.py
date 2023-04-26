"""
Create dataset object from json.
Split into train, validation and test set.
Add sentence and label
Add tokenization column
Convert to tf_dataset
"""

from datasets import load_dataset

class TfDataset:
    def __init__(self, json_data_file):

        self.test_ds = load_dataset("json", data_files=json_data_file, split="train[90%:]")
        self.val_ds = load_dataset("json", data_files=json_data_file,
                              split=[f"train[{k}%:{k + 9}%]" for k in range(0, 90, 9)])
        self.train_ds = load_dataset("json", data_files=json_data_file,
                                split=[f"train[:{k}%]+train[{k + 9}%:]" for k in range(0, 90, 9)])


    def testing(self):
        pass
