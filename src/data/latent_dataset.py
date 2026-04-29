"""Dataset for precomputed latent vectors.

Loads a ``.pt`` file produced by the export_latents pipeline and exposes
each sample as a dict of tensors.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


class LatentDataset(Dataset):
    """PyTorch Dataset that wraps a precomputed latent .pt file.

    The file is expected to contain a list of dicts, each with:
        - ``z_normalized``   : ``(num_latent_tokens, latent_dim)``
        - ``context_ids``    : ``(max_context_len,)``
        - ``context_mask``   : ``(max_context_len,)``
        - ``question_ids``   : ``(max_question_len,)``
        - ``question_mask``  : ``(max_question_len,)``
        - ``is_answerable``  : scalar bool/float tensor

    Parameters
    ----------
    latent_dir : str
        Directory containing the latent dataset files.
    split : str
        Dataset split (``"train"`` or ``"val"``).  The file loaded is
        ``{latent_dir}/latent_dataset_{split}.pt``.
    """

    _REQUIRED_KEYS = {
        "z_normalized",
        "context_ids",
        "context_mask",
        "question_ids",
        "question_mask",
        "is_answerable",
    }

    def __init__(self, latent_dir: str, split: str) -> None:
        path = Path(latent_dir) / f"latent_dataset_{split}.pt"
        raw = torch.load(path, map_location="cpu", weights_only=True)

        # Accept either a list of dicts or a single dict of stacked tensors.
        if isinstance(raw, list):
            # Legacy list-of-dicts format — validate keys on first sample.
            if len(raw) == 0:
                raise ValueError(f"Latent dataset at '{path}' is empty.")
            missing = self._REQUIRED_KEYS - set(raw[0].keys())
            if missing:
                raise ValueError(
                    f"Latent dataset is missing required keys: {sorted(missing)}"
                )
            self._stacked: dict[str, Tensor] | None = None
            self._list: list[dict[str, Tensor]] = raw
            self._n: int = len(raw)
        elif isinstance(raw, dict):
            # Dict-of-tensors format: each value has shape (N, ...).
            # Index on-the-fly to avoid materialising N individual dicts.
            if len(raw) == 0 or len(next(iter(raw.values()))) == 0:
                raise ValueError(f"Latent dataset at '{path}' is empty.")
            missing = self._REQUIRED_KEYS - set(raw.keys())
            if missing:
                raise ValueError(
                    f"Latent dataset is missing required keys: {sorted(missing)}"
                )
            self._stacked = raw
            self._list = []
            self._n = len(next(iter(raw.values())))
        else:
            raise ValueError(
                f"Unsupported latent dataset format: {type(raw)}.  "
                "Expected a list of dicts or a dict of tensors."
            )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        if self._stacked is not None:
            return {k: self._stacked[k][idx] for k in self._stacked}
        sample = self._list[idx]
        return {k: v for k, v in sample.items()}
