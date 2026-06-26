# extract_saprot.py
# -*- coding: utf-8 -*-

from contextlib import nullcontext

import torch
import numpy as np
from transformers import EsmTokenizer, EsmModel


class SaProtExtractor:
    def __init__(
        self,
        model_name="westlake-repl/SaProt_1.3B_AF2",
        max_length=257,
        use_8bit=False,
        device=None,
    ):
        self.model_name = model_name
        self.max_length = int(max_length)
        self.max_residues = self.max_length - 2
        self.use_8bit = bool(use_8bit)

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = EsmTokenizer.from_pretrained(self.model_name)

        if self.use_8bit:
            self.model = EsmModel.from_pretrained(
                self.model_name,
                load_in_8bit=True,
                device_map="auto",
            )
        else:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.model = EsmModel.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
            ).to(self.device).eval()

        self.hidden_size = int(self.model.config.hidden_size)

    @staticmethod
    def sanitize_aa(sequence: str) -> str:
        """
        Keep consistent with the training stage:
        Allow A C D E F G H I K L M N P Q R S T V W Y B X Z U O J;
        Replace other characters with X.
        """
        aa_allowed = set("ACDEFGHIKLMNPQRSTVWYBXZUOJ")
        sequence = sequence.strip().upper()
        sequence = "".join([c if c in aa_allowed else "X" for c in sequence])

        if len(sequence) == 0:
            raise ValueError("Input sequence is empty.")

        return sequence

    @staticmethod
    def aa_to_saprot_input(aa_seq: str) -> str:
        """
        SaProt expects each position to be AA + structure token.
        Use # as a placeholder when there is no structure token.
        Example: ACD -> A#C#D#
        """
        return "".join([aa + "#" for aa in aa_seq])

    def mean_pool_last_hidden(self, last_hidden, input_ids, attention_mask):
        """
        Keep consistent with the training stage:
        After removing cls/eos/pad, perform masked mean pooling on valid tokens.
        """
        attn = attention_mask.bool()

        special = torch.zeros_like(attn)

        if self.tokenizer.cls_token_id is not None:
            special |= (input_ids == self.tokenizer.cls_token_id)

        if self.tokenizer.eos_token_id is not None:
            special |= (input_ids == self.tokenizer.eos_token_id)

        if self.tokenizer.pad_token_id is not None:
            special |= (input_ids == self.tokenizer.pad_token_id)

        mask = attn & (~special)
        mask_f = mask.unsqueeze(-1).type_as(last_hidden)

        pooled = (last_hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)
        return pooled

    def extract_one(self, sequence: str) -> np.ndarray:
        """
        Input:
            sequence: Raw amino acid sequence, e.g. KWKLFKKIGAVLKVL
        Output:
            feature: shape = (1, hidden_size), usually (1, 1280), dtype=np.float32
        """
        aa_seq = self.sanitize_aa(sequence)

        if len(aa_seq) > self.max_residues:
            aa_seq = aa_seq[:self.max_residues]

        saprot_seq = self.aa_to_saprot_input(aa_seq)

        inputs = self.tokenizer(
            [saprot_seq],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        if not self.use_8bit:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        use_amp = (self.device.type == "cuda") and (not self.use_8bit)
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
        amp_ctx = torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()

        with torch.no_grad():
            with amp_ctx:
                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state

            pooled = self.mean_pool_last_hidden(
                last_hidden,
                inputs["input_ids"],
                inputs["attention_mask"],
            )

        feature = pooled.float().cpu().numpy().astype(np.float32)

        if feature.shape[1] != self.hidden_size:
            raise ValueError(
                f"SaProt feature dimension anomaly: feature.shape={feature.shape}, "
                f"hidden_size={self.hidden_size}"
            )

        return feature

    def extract_batch(self, sequences, batch_size=1) -> np.ndarray:
        """
        Extract SaProt features in batch.
        Consistent with the training stage, default batch_size=1.
        """
        features = []

        use_amp = (self.device.type == "cuda") and (not self.use_8bit)
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
        amp_ctx = torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()

        cleaned = []
        for seq in sequences:
            aa_seq = self.sanitize_aa(seq)
            if len(aa_seq) > self.max_residues:
                aa_seq = aa_seq[:self.max_residues]
            cleaned.append(aa_seq)

        with torch.no_grad():
            for i in range(0, len(cleaned), batch_size):
                batch_aa = cleaned[i:i + batch_size]
                batch_saprot = [self.aa_to_saprot_input(s) for s in batch_aa]

                inputs = self.tokenizer(
                    batch_saprot,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )

                if not self.use_8bit:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with amp_ctx:
                    outputs = self.model(**inputs)
                    last_hidden = outputs.last_hidden_state

                pooled = self.mean_pool_last_hidden(
                    last_hidden,
                    inputs["input_ids"],
                    inputs["attention_mask"],
                )

                features.append(pooled.float().cpu().numpy().astype(np.float32))

        features = np.concatenate(features, axis=0).astype(np.float32)

        if features.shape[1] != self.hidden_size:
            raise ValueError(
                f"SaProt feature dimension anomaly: features.shape={features.shape}, "
                f"hidden_size={self.hidden_size}"
            )

        return features