"""Tests for config schema dataclasses."""

import dataclasses
from typing import Any

import pytest

from src.config.schema import Config


class TestConfigFrozen:
    def test_config_is_frozen(self, tiny_config):
        with pytest.raises(dataclasses.FrozenInstanceError):
            tiny_config.seed = 99

    def test_nested_config_is_frozen(self, tiny_config):
        with pytest.raises(dataclasses.FrozenInstanceError):
            tiny_config.vae_arch.latent_dim = 999


class TestConfigRoundTrip:
    def test_to_dict_round_trip(self, tiny_config):
        d = tiny_config.to_dict()
        restored = Config.from_dict(d)
        assert restored == tiny_config

    def test_to_dict_returns_dict(self, tiny_config):
        d = tiny_config.to_dict()
        assert isinstance(d, dict)
        assert "seed" in d
        assert "vae_arch" in d


class TestFieldTypes:
    def test_all_fields_have_types(self):
        for section_field in dataclasses.fields(Config):
            if dataclasses.is_dataclass(section_field.type):
                cls = section_field.type
            else:
                # Resolve string annotations for nested dataclasses
                continue
            for f in dataclasses.fields(cls):
                assert f.type is not Any, (
                    f"{cls.__name__}.{f.name} has type Any"
                )

    def test_default_config_creates(self):
        config = Config()
        assert config.seed == 42
