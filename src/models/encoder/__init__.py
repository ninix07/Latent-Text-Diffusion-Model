"""Encoder subpackage: frozen pretrained encoder and conditioning projection."""

from src.models.encoder.frozen_encoder import FrozenEncoder
from src.models.encoder.projection import ConditioningProjection

__all__ = ["FrozenEncoder", "ConditioningProjection"]
