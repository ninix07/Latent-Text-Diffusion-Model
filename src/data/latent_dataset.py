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
        - ``z_normalized``   : ``(max_answer_len, latent_dim)``
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
        raw = torch.load(path, map_location="cpu", weights_only=False)

        # Accept either a list of dicts or a single dict of lists/tensors.
        if isinstance(raw, list):
            self._data: list[dict[str, Tensor]] = raw
        elif isinstance(raw, dict):
            # Dict-of-tensors format: each value has shape (N, ...)
            n = len(next(iter(raw.values())))
            self._data = [{k: raw[k][i] for k in raw} for i in range(n)]
        else:
            raise ValueError(
                f"Unsupported latent dataset format: {type(raw)}.  "
                "Expected a list of dicts or a dict of tensors."
            )

        if len(self._data) == 0:
            raise ValueError(f"Latent dataset at '{path}' is empty.")

        # Validate keys on the first sample
        first_keys = set(self._data[0].keys())
        missing = self._REQUIRED_KEYS - first_keys
        if missing:
            raise ValueError(
                f"Latent dataset is missing required keys: {sorted(missing)}"
            )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        sample = self._data[idx]
        # Return a shallow copy so callers cannot mutate the stored data.
        return {k: v for k, v in sample.items()}
