import json
import unittest
from typing import List, Dict, Any
import random
from random import choices

import numpy as np
import torch
from torch import nn

from lr_scheduler import NoamOpt
from transformer import Transformer
from vocabulary import Vocabulary
from utils import construct_batches, construct_future_mask
from transformer_dataset import TransformerDataset


def train(
        transformer: nn.Module,
        scheduler: Any,
        criterion: Any,
        batches: Dict[str, List[torch.Tensor]],
        masks: Dict[str, List[torch.Tensor]],
        n_epochs: int,
):
    """
    Main training loop

    :param transformer: the transformer model
    :param scheduler: the learning rate scheduler
    :param criterion: the optimization criterion (loss function)
    :param batches: aligned src and tgt batches that contain tokens ids
    :param masks: source key padding mask and target future mask for each batch
    :param n_epochs: the number of epochs to train the model for
    :return: the accuracy and loss on the latest batch
    """
    transformer.train(True)
    num_iters = 0

    for e in range(n_epochs):
        for i, (src_batch, src_mask, tgt_batch, tgt_mask) in enumerate(
                zip(batches["src"], masks["src"], batches["tgt"], masks["tgt"])
        ):
            encoder_output = transformer.encoder(src_batch, src_padding_mask=src_mask)  # type: ignore

            # Perform one decoder forward pass to obtain *all* next-token predictions for every index i given its
            # previous *gold standard* tokens [1,..., i] (i.e. teacher forcing) in parallel/at once.
            decoder_output = transformer.decoder(
                tgt_batch,
                encoder_output,
                src_padding_mask=src_mask,
                future_mask=tgt_mask,
            )  # type: ignore

            # Align labels with predictions: the last decoder prediction is meaningless because we have no target token
            # for it. The BOS token in the target is also not something we want to compute a loss for.
            decoder_output = decoder_output[:, :-1, :]
            tgt_batch = tgt_batch[:, 1:]

            # Set pad tokens in the target to -100 so they don't incur a loss
            # tgt_batch[tgt_batch == transformer.padding_idx] = -100

            # Compute the average cross-entropy loss over all next-token predictions at each index i given [1, ..., i]
            # for the entire batch. Note that the original paper uses label smoothing (I was too lazy).
            batch_loss = criterion(
                decoder_output.contiguous().permute(0, 2, 1),
                tgt_batch.contiguous().long(),
            )

            # Rough estimate of per-token accuracy in the current training batch
            batch_accuracy = (
                                 torch.sum(decoder_output.argmax(dim=-1) == tgt_batch)
                             ) / torch.numel(tgt_batch)

            if num_iters % 100 == 0:
                print(
                    f"epoch: {e}, num_iters: {num_iters}, batch_loss: {batch_loss}, batch_accuracy: {batch_accuracy}"
                )

            # Update parameters
            batch_loss.backward()
            scheduler.step()
            scheduler.optimizer.zero_grad()
            num_iters += 1
    return batch_loss, batch_accuracy


class Test:

    def __init__(self):
        self.src_corpus = []
        self.tgt_corpus = []
        self.test_data = []

    def Train(self):
        seed = 0
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        """
        Test training by trying to (over)fit a simple copy dataset - bringing the loss to ~zero. (GPU required)
        """
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # if device.type == "cpu":
        #     print("This unit test was not run because it requires a GPU")
        #     return

        # Hyperparameters
        # synthetic_corpus_size = 600
        # batch_size = 60
        # n_epochs = 200
        # n_tokens_in_batch = 10
        # synthetic_corpus_size = 10
        batch_size = 60
        n_epochs = 2
        # n_tokens_in_batch = 10

        train_file_name = "../../data/en-train.conll"
        dev_file_name = "../../data/en-dev.conll"

        formatted_train_file = "../data/train_data.json"
        formatted_dev_file = "../data/dev_data.json"
        formatted_test_file = "../data/test_data.json"

        tran_dataset = TransformerDataset(
            train_file_name,
            dev_file_name,
            formatted_train_file,
            formatted_dev_file,
            formatted_test_file
        )
        train_data = tran_dataset.readData(train=True)

        # to make the vocab on whole dataset
        self.test_data = tran_dataset.readData()

        data = train_data + self.test_data

        self.src_corpus = []
        self.tgt_corpus = []
        for value in data:
            self.src_corpus.append(value["sentence"])
            self.tgt_corpus.append(value["tag"])

        # print(src_corpus[0])
        # print(tgt_corpus[0])

        # Construct vocabulary and create synthetic data by uniform randomly sampling tokens from it
        # Note: the original paper uses byte pair encodings, we simply take each word to be a token.
        # corpus = ["These are the tokens that will end up in our vocabulary"]
        src_vocab = Vocabulary(self.src_corpus)
        tgt_vocab = Vocabulary(self.tgt_corpus)

        src_vocab_size = len(
            list(src_vocab.token2index.keys())
        )  # 14 tokens including bos, eos and pad

        tgt_vocab_size = len(
            list(tgt_vocab.token2index.keys())
        )  # 14 tokens including bos, eos and pad

        # valid_tokens = list(src_vocab.token2index.keys())[3:]

        # add repeated same sentence with shuffled words
        # corpus += [
        #     " ".join(choices(valid_tokens, k=n_tokens_in_batch))
        #     for _ in range(synthetic_corpus_size)
        # ]

        # Construct src-tgt aligned input batches (note: the original paper uses dynamic batching based on tokens)
        corpus = [{"src": sent["sentence"], "tgt": sent["tag"]} for sent in train_data]
        # print(corpus)
        batches, masks = construct_batches(
            corpus,
            src_vocab,
            tgt_vocab,
            batch_size=batch_size,
            src_lang_key="src",
            tgt_lang_key="tgt",
            device=device,
        )

        # Initialize transformer
        transformer = Transformer(
            hidden_dim=512,
            ff_dim=2048,
            num_heads=8,
            num_layers=2,
            max_decoding_length=25,
            encoder_vocab_size=src_vocab_size,
            decoder_vocab_size=tgt_vocab_size,
            encoder_padding_idx=src_vocab.token2index[src_vocab.PAD],
            decoder_padding_idx=tgt_vocab.token2index[tgt_vocab.PAD],
            encoder_bos_idx=src_vocab.token2index[src_vocab.BOS],
            decoder_bos_idx=tgt_vocab.token2index[tgt_vocab.BOS],
            dropout_p=0.1,
            tie_output_to_embedding=True,
        ).to(device)

        # Initialize learning rate scheduler, optimizer and loss (note: the original paper uses label smoothing)
        optimizer = torch.optim.Adam(
            transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
        )
        scheduler = NoamOpt(
            transformer.hidden_dim, factor=1, warmup=400, optimizer=optimizer,
        )
        criterion = nn.CrossEntropyLoss()

        # Start training and verify ~zero loss and >90% accuracy on the last batch
        latest_batch_loss, latest_batch_accuracy = train(
            transformer, scheduler, criterion, batches, masks, n_epochs=n_epochs
        )

    def Inference(self):

        en_vocab = Vocabulary(self.src_corpus)
        en_vocab_size = len(en_vocab.token2index.items())

        de_vocab = Vocabulary(self.tgt_corpus)
        de_vocab_size = len(de_vocab.token2index.items())
        with torch.no_grad():
            transformer = Transformer(
                hidden_dim=512,
                ff_dim=2048,
                num_heads=8,
                num_layers=6,
                max_decoding_length=10,
                encoder_vocab_size=en_vocab_size,
                decoder_vocab_size=de_vocab_size,
                encoder_padding_idx=en_vocab.token2index[en_vocab.PAD],
                decoder_padding_idx=de_vocab.token2index[de_vocab.PAD],
                encoder_bos_idx=en_vocab.token2index[en_vocab.BOS],
                decoder_bos_idx=de_vocab.token2index[de_vocab.BOS],
                dropout_p=0.1,
                tie_output_to_embedding=True,
            )
            transformer.eval()

            en_corpus = []
            # de_corpus = []
            for value in self.test_data:
                en_corpus.append(value["sentence"])
                # de_corpus.append(value["tag"])

            # Prepare encoder input, mask and generate output hidden states
            encoder_input = torch.IntTensor(
                en_vocab.batch_encode(en_corpus, add_special_tokens=False)
            )
            src_padding_mask = encoder_input != transformer.encoder_padding_idx
            encoder_output = transformer.encoder.forward(
                encoder_input, src_padding_mask=src_padding_mask
            )
            # self.assertEqual(torch.any(torch.isnan(encoder_output)), False)

            # Prepare decoder input and mask and start decoding
            decoder_input = torch.IntTensor(
                [[transformer.encoder_bos_idx], [transformer.encoder_bos_idx]]
            )
            future_mask = construct_future_mask(seq_len=1)
            for i in range(transformer.max_decoding_length):
                decoder_output = transformer.decoder(
                    decoder_input,
                    encoder_output,
                    src_padding_mask=src_padding_mask,
                    future_mask=future_mask,
                )

                # Take the argmax over the softmax of the last token to obtain the next-token prediction
                predicted_tokens = torch.argmax(
                    decoder_output[:, -1, :], dim=-1
                ).unsqueeze(1)

                # Append the prediction to the already decoded tokens and construct the new mask
                decoder_input = torch.cat((decoder_input, predicted_tokens), dim=-1)
                future_mask = construct_future_mask(decoder_input.shape[1])

        predicted_tokens = decoder_input.tolist()

        # Compare the predicted and expected tokens for each instance
        for i in range(len(predicted_tokens)):
            predicted = predicted_tokens[i]
            expected = self.tgt_corpus[i]

            if predicted == expected:
                print(f"Inference correct for instance {i}")
            else:
                print(f"Inference incorrect for instance {i}")
                print("Predicted:", predicted)
                print("Expected:", expected)

        # self.assertEqual(decoder_input.shape, (2, transformer.max_decoding_length + 1))
        # see test_one_layer_transformer_decoder_inference in decoder.py for more information. with num_layers=1 this
        # will be true.
        # self.assertEqual(torch.all(decoder_input == transformer.decoder_bos_idx), False)

class TestTransformerTraining(unittest.TestCase):
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    def test_copy_task(self):
        """
        Test training by trying to (over)fit a simple copy dataset - bringing the loss to ~zero. (GPU required)
        """
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # if device.type == "cpu":
        #     print("This unit test was not run because it requires a GPU")
        #     return

        # Hyperparameters
        # synthetic_corpus_size = 600
        # batch_size = 60
        # n_epochs = 200
        # n_tokens_in_batch = 10
        synthetic_corpus_size = 10
        batch_size = 60
        n_epochs = 2
        n_tokens_in_batch = 10

        # Construct vocabulary and create synthetic data by uniform randomly sampling tokens from it
        # Note: the original paper uses byte pair encodings, we simply take each word to be a token.
        corpus = ["These are the tokens that will end up in our vocabulary"]
        vocab = Vocabulary(corpus)
        vocab_size = len(
            list(vocab.token2index.keys())
        )  # 14 tokens including bos, eos and pad

        valid_tokens = list(vocab.token2index.keys())[3:]

        corpus += [
            " ".join(choices(valid_tokens, k=n_tokens_in_batch))
            for _ in range(synthetic_corpus_size)
        ]
        # print(corpus)

        # Construct src-tgt aligned input batches (note: the original paper uses dynamic batching based on tokens)
        corpus = [{"src": sent, "tgt": sent} for sent in corpus]
        print(corpus)
        batches, masks = construct_batches(
            corpus,
            vocab,
            vocab,
            batch_size=batch_size,
            src_lang_key="src",
            tgt_lang_key="tgt",
            device=device,
        )

        # Initialize transformer
        transformer = Transformer(
            hidden_dim=512,
            ff_dim=2048,
            num_heads=8,
            num_layers=2,
            max_decoding_length=25,
            encoder_vocab_size=vocab_size,
            decoder_vocab_size=vocab_size,
            encoder_padding_idx=vocab.token2index[vocab.PAD],
            decoder_padding_idx=vocab.token2index[vocab.PAD],
            encoder_bos_idx=vocab.token2index[vocab.BOS],
            decoder_bos_idx=vocab.token2index[vocab.BOS],
            dropout_p=0.1,
            tie_output_to_embedding=True,
        ).to(device)

        # Initialize learning rate scheduler, optimizer and loss (note: the original paper uses label smoothing)
        optimizer = torch.optim.Adam(
            transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
        )
        scheduler = NoamOpt(
            transformer.hidden_dim, factor=1, warmup=400, optimizer=optimizer,
        )
        criterion = nn.CrossEntropyLoss()

        # Start training and verify ~zero loss and >90% accuracy on the last batch
        latest_batch_loss, latest_batch_accuracy = train(
            transformer, scheduler, criterion, batches, masks, n_epochs=n_epochs
        )
        self.assertEqual(latest_batch_loss.item() <= 0.01, True)
        self.assertEqual(latest_batch_accuracy >= 0.99, True)


if __name__ == "__main__":
    unittest.main()
