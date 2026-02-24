"""Unit tests for ModelSpec, ModelRegistry, and TPHandlers — CPU only."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cuda_ipc_poc.model import SimpleMLP
from cuda_ipc_poc.model_spec import ModelRegistry, ModelSpec
from cuda_ipc_poc.tp_handler import BaseTensorParallelHandler, SimpleMlpTPHandler
from cuda_ipc_poc.tensor_parallel import shard_model


class TestModelRegistry:
    def test_register_and_get(self):
        spec = ModelRegistry.get("mlp")
        assert spec.model_cls is SimpleMLP

    def test_list_models_includes_mlp(self):
        models = ModelRegistry.list_models()
        assert "mlp" in models
        assert "resnet18" in models

    def test_get_unknown_raises_keyerror(self):
        with pytest.raises(KeyError, match="Unknown model"):
            ModelRegistry.get("nonexistent-model")

    def test_factory_creates_correct_model(self):
        spec = ModelRegistry.get("mlp")
        model = spec.model_factory()
        assert isinstance(model, SimpleMLP)

    def test_sample_input_fn(self):
        spec = ModelRegistry.get("mlp")
        x = spec.sample_input_fn("cpu", batch_size=2)
        assert x.shape == (2, 784)
        assert x.device == torch.device("cpu")


class TestBaseTensorParallelHandler:
    def test_unmatched_params_replicated(self):
        handler = BaseTensorParallelHandler()
        state = {"a.weight": torch.randn(4, 4), "b.bias": torch.randn(4)}
        result = handler.process_state_dict(state, world_size=2, rank=0)
        assert torch.equal(result["a.weight"], state["a.weight"])
        assert torch.equal(result["b.bias"], state["b.bias"])

    def test_column_wise_splits_dim0(self):
        handler = BaseTensorParallelHandler()
        handler.COLUMN_WISE_PARAMS = ["w"]
        state = {"w": torch.randn(8, 4)}
        r0 = handler.process_state_dict(state, world_size=2, rank=0)
        r1 = handler.process_state_dict(state, world_size=2, rank=1)
        assert r0["w"].shape == (4, 4)
        assert r1["w"].shape == (4, 4)
        assert torch.equal(torch.cat([r0["w"], r1["w"]], dim=0), state["w"])

    def test_row_wise_splits_dim1(self):
        handler = BaseTensorParallelHandler()
        handler.ROW_WISE_PARAMS = ["w"]
        state = {"w": torch.randn(4, 8)}
        r0 = handler.process_state_dict(state, world_size=2, rank=0)
        r1 = handler.process_state_dict(state, world_size=2, rank=1)
        assert r0["w"].shape == (4, 4)
        assert r1["w"].shape == (4, 4)
        assert torch.equal(torch.cat([r0["w"], r1["w"]], dim=1), state["w"])

    def test_glob_pattern_matching(self):
        handler = BaseTensorParallelHandler()
        handler.COLUMN_WISE_PARAMS = ["layers.*.attn.q_proj.weight"]
        state = {
            "layers.0.attn.q_proj.weight": torch.randn(8, 4),
            "layers.1.attn.q_proj.weight": torch.randn(8, 4),
            "layers.0.norm.weight": torch.randn(4),
        }
        result = handler.process_state_dict(state, world_size=2, rank=0)
        assert result["layers.0.attn.q_proj.weight"].shape == (4, 4)
        assert result["layers.1.attn.q_proj.weight"].shape == (4, 4)
        # norm.weight should be replicated (no pattern match)
        assert result["layers.0.norm.weight"].shape == (4,)

    def test_contiguity(self):
        handler = BaseTensorParallelHandler()
        handler.COLUMN_WISE_PARAMS = ["w"]
        state = {"w": torch.randn(8, 4)}
        r0 = handler.process_state_dict(state, world_size=2, rank=0)
        assert r0["w"].is_contiguous()


class TestSimpleMlpTPHandler:
    @pytest.fixture
    def model(self):
        torch.manual_seed(0)
        return SimpleMLP()

    def test_parity_with_shard_model(self, model):
        """SimpleMlpTPHandler must produce identical results to legacy shard_model()."""
        handler = SimpleMlpTPHandler()
        state = model.state_dict()

        for world_size in [2, 4]:
            for rank in range(world_size):
                legacy = shard_model(model, world_size, rank)
                new = handler.process_state_dict(state, world_size, rank)
                assert set(legacy.keys()) == set(new.keys()), (
                    f"ws={world_size} rank={rank}: key mismatch"
                )
                for k in legacy:
                    assert torch.equal(legacy[k], new[k]), (
                        f"ws={world_size} rank={rank} key={k}: tensor mismatch"
                    )

    def test_fc2_bias_only_rank_0(self, model):
        handler = SimpleMlpTPHandler()
        state = model.state_dict()
        r0 = handler.process_state_dict(state, world_size=2, rank=0)
        r1 = handler.process_state_dict(state, world_size=2, rank=1)
        assert "fc2.bias" in r0
        assert "fc2.bias" not in r1

    def test_shard_shapes(self, model):
        handler = SimpleMlpTPHandler()
        state = model.state_dict()
        shard = handler.process_state_dict(state, world_size=2, rank=0)
        assert shard["fc1.weight"].shape == (128, 784)
        assert shard["fc1.bias"].shape == (128,)
        assert shard["fc2.weight"].shape == (10, 128)
        assert shard["fc2.bias"].shape == (10,)
