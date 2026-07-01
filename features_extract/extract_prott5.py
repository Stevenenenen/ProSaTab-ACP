# extract_prott5.py
# -*- coding: utf-8 -*-

import re
from pathlib import Path

import torch
import numpy as np
from transformers import T5Model, T5Tokenizer


class ProtT5Extractor:
    def __init__(
        self,
        model_path="./local/prott5",
        max_length=255,
        device=None,
    ):
        self.model_path = Path(model_path)
        self.max_length = int(max_length)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if not self.model_path.exists():
            raise FileNotFoundError(f"ProtT5 model path not found: {self.model_path}")

        self.model = T5Model.from_pretrained(
            str(self.model_path)
        ).to(self.device)

        self.tokenizer = T5Tokenizer.from_pretrained(
            str(self.model_path),
            legacy=False
        )

        self.model.eval()

    @staticmethod
    def clean_sequence(sequence: str) -> str:
        sequence = sequence.strip().upper()
        sequence = re.sub(r"\s+", "", sequence)

        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        invalid = set(sequence) - valid_aa

        if len(sequence) == 0:
            raise ValueError("Empty sequence.")

        if invalid:
            raise ValueError(f"Invalid amino acid characters: {invalid}")

        return " ".join(sequence)

    def extract_one(self, sequence: str) -> np.ndarray:
        processed_seq = self.clean_sequence(sequence)

        batch_tokens = self.tokenizer(
            [processed_seq],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        input_ids = batch_tokens["input_ids"]
        attention_mask = batch_tokens["attention_mask"]

        with torch.no_grad():
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Same pooling strategy as training.
            feature = encoder_outputs.last_hidden_state.mean(dim=1)

        feature = feature.cpu().numpy().astype(np.float32)

        if feature.shape[1] != 1024:
            raise ValueError(f"Unexpected ProtT5 feature shape: {feature.shape}")

        return feature

    def extract_batch(self, sequences, batch_size=8) -> np.ndarray:
        processed_sequences = [self.clean_sequence(seq) for seq in sequences]

        all_features = []

        for i in range(0, len(processed_sequences), batch_size):
            batch_sequences = processed_sequences[i:i + batch_size]

            batch_tokens = self.tokenizer(
                batch_sequences,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            input_ids = batch_tokens["input_ids"]
            attention_mask = batch_tokens["attention_mask"]

            with torch.no_grad():
                encoder_outputs = self.model.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                batch_features = encoder_outputs.last_hidden_state.mean(dim=1)

            all_features.append(batch_features.cpu().numpy().astype(np.float32))

        features = np.vstack(all_features).astype(np.float32)

        if features.shape[1] != 1024:
            raise ValueError(f"Unexpected ProtT5 feature shape: {features.shape}")

        return features