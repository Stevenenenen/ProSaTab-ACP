# extract_saprot_prott5.py
# -*- coding: utf-8 -*-

"""
SaProt + ProtT5 Joint Feature Extractor

Keep consistent with the training stage:
1. SaProt features first
2. ProtT5 features second
3. Concatenation: np.concatenate([saprot_feat, prott5_feat], axis=1)
4. Output dimension: 1280 + 1024 = 2304

Training stage corresponding code:
X1 = SaProt features
X2 = ProtT5 features
X = np.concatenate([X1, X2], axis=1)
"""

import numpy as np

from .extract_saprot import SaProtExtractor
from .extract_prott5 import ProtT5Extractor


class SaProtProtT5Extractor:
    def __init__(
        self,
        saprot_model_name: str = "westlake-repl/SaProt_1.3B_AF2",
        prott5_model_path: str = "./local/prott5",
        device=None,
        expected_saprot_dim: int = 1280,
        expected_prott5_dim: int = 1024,
        expected_total_dim: int = 2304,
    ):
        self.expected_saprot_dim = int(expected_saprot_dim)
        self.expected_prott5_dim = int(expected_prott5_dim)
        self.expected_total_dim = int(expected_total_dim)

        self.saprot = SaProtExtractor(
            model_name=saprot_model_name,
            max_length=257,
            use_8bit=False,
            device=device,
        )

        self.prott5 = ProtT5Extractor(
            model_path=prott5_model_path,
            max_length=255,
            device=device,
        )

    def extract_one(self, sequence: str) -> np.ndarray:
        """
        Input:
            sequence: Raw amino acid sequence, e.g. "KWKLFKKIGAVLKVL"

        Output:
            feature: shape = (1, 2304), dtype = np.float32

        Concatenation order:
            [SaProt, ProtT5]
        """
        saprot_feat = self.saprot.extract_one(sequence)
        prott5_feat = self.prott5.extract_one(sequence)

        self._check_component_shape(
            saprot_feat=saprot_feat,
            prott5_feat=prott5_feat,
        )

        feature = np.concatenate(
            [saprot_feat, prott5_feat],
            axis=1,
        ).astype(np.float32)

        self._check_total_shape(feature)

        return feature

    def extract_batch(
        self,
        sequences,
        batch_size_saprot: int = 1,
        batch_size_prott5: int = 8,
    ) -> np.ndarray:
        """
        Extract joint features in batch.

        Input:
            sequences: list[str]

        Output:
            feature: shape = (N, 2304), dtype = np.float32

        Concatenation order:
            [SaProt, ProtT5]
        """
        saprot_feat = self.saprot.extract_batch(
            sequences,
            batch_size=batch_size_saprot,
        )

        prott5_feat = self.prott5.extract_batch(
            sequences,
            batch_size=batch_size_prott5,
        )

        if saprot_feat.shape[0] != prott5_feat.shape[0]:
            raise ValueError(
                "SaProt and ProtT5 sample counts do not match, cannot concatenate: "
                f"SaProt={saprot_feat.shape}, ProtT5={prott5_feat.shape}"
            )

        self._check_component_shape(
            saprot_feat=saprot_feat,
            prott5_feat=prott5_feat,
        )

        feature = np.concatenate(
            [saprot_feat, prott5_feat],
            axis=1,
        ).astype(np.float32)

        self._check_total_shape(feature)

        return feature

    def _check_component_shape(
        self,
        saprot_feat: np.ndarray,
        prott5_feat: np.ndarray,
    ):
        if saprot_feat.ndim != 2:
            raise ValueError(
                f"SaProt feature should be a 2D matrix, current shape={saprot_feat.shape}"
            )

        if prott5_feat.ndim != 2:
            raise ValueError(
                f"ProtT5 feature should be a 2D matrix, current shape={prott5_feat.shape}"
            )

        if saprot_feat.shape[1] != self.expected_saprot_dim:
            raise ValueError(
                "SaProt feature dimension anomaly: "
                f"current {saprot_feat.shape[1]}, expected {self.expected_saprot_dim}"
            )

        if prott5_feat.shape[1] != self.expected_prott5_dim:
            raise ValueError(
                "ProtT5 feature dimension anomaly: "
                f"current {prott5_feat.shape[1]}, expected {self.expected_prott5_dim}"
            )

    def _check_total_shape(self, feature: np.ndarray):
        if feature.ndim != 2:
            raise ValueError(
                f"Joint feature should be a 2D matrix, current shape={feature.shape}"
            )

        if feature.shape[1] != self.expected_total_dim:
            raise ValueError(
                "Joint feature dimension anomaly: "
                f"current {feature.shape[1]}, expected {self.expected_total_dim}. "
                "Please check SaProt/ProtT5 dimensions or concatenation order."
            )


def extract_saprot_prott5_feature(
    sequence: str,
    saprot_model_name: str = "westlake-repl/SaProt_1.3B_AF2",
    prott5_model_path: str = "./local/prott5",
    device=None,
) -> np.ndarray:
    """
    Convenience function: extract SaProt-ProtT5 joint feature for a single sequence.

    Output:
        shape = (1, 2304)
    """
    extractor = SaProtProtT5Extractor(
        saprot_model_name=saprot_model_name,
        prott5_model_path=prott5_model_path,
        device=device,
    )

    return extractor.extract_one(sequence)


# # This is a simple test
# if __name__ == "__main__":
#     seq = "KWKLFKKIGAVLKVL"
#
#     extractor = SaProtProtT5Extractor(
#         saprot_model_name="westlake-repl/SaProt_1.3B_AF2",
#         prott5_model_path="./local/prott5",
#         device=None,
#     )
#
#     X = extractor.extract_one(seq)
#
#     print("Joint feature shape:", X.shape)
#     print("Joint feature dtype:", X.dtype)