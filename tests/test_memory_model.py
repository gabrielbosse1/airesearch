"""DualMemory: 1024 keys / 8192 values, FIFO, detachment, persistence."""
import pytest
import torch

from models.memory_model import DualMemory, ProceduralSkillBank, SkillBankView


def test_production_geometry_defaults():
    bank = ProceduralSkillBank(capacity=4)
    assert bank.key_dim == 1024 and bank.value_dim == 8192
    assert bank.keys.shape == (4, 1024)
    assert bank.values.shape == (4, 8192)


def test_write_and_view_small():
    bank = ProceduralSkillBank(capacity=3, key_dim=8, value_dim=16)
    key, value = torch.randn(8), torch.randn(16)
    slot = bank.write(key, value, modality="image", metadata={"tag": "x"})
    assert slot == 0 and len(bank) == 1
    view = bank.view()
    assert view.keys.shape == (1, 8) and view.values.shape == (1, 16)
    assert torch.equal(view.values[0], value)
    assert bank.metadata[0]["modality"] == "image"
    assert bank.metadata[0]["tag"] == "x"


def test_writes_are_detached_storage_events():
    bank = ProceduralSkillBank(capacity=2, key_dim=8, value_dim=16)
    value = torch.randn(16, requires_grad=True)
    key = torch.randn(8, requires_grad=True)
    bank.write(key, value)
    assert not bank.values.requires_grad
    assert not bank.keys.requires_grad
    # The stored copy must not retain the producer's graph.
    assert bank.values.grad_fn is None


def test_fifo_eviction_is_deterministic():
    bank = ProceduralSkillBank(capacity=2, key_dim=4, value_dim=4)
    bank.write(torch.zeros(4), torch.full((4,), 1.0))
    bank.write(torch.zeros(4), torch.full((4,), 2.0))
    bank.write(torch.zeros(4), torch.full((4,), 3.0))  # evicts slot 0 (oldest)
    assert len(bank) == 2
    values = {float(v[0]) for v in bank.values}
    assert values == {2.0, 3.0}


def test_invalid_writes_rejected():
    bank = ProceduralSkillBank(capacity=2, key_dim=8, value_dim=16)
    with pytest.raises(ValueError):
        bank.write(torch.randn(7), torch.randn(16))
    with pytest.raises(ValueError):
        bank.write(torch.randn(8), torch.randn(15))
    with pytest.raises(ValueError):
        bank.write(torch.randn(8), torch.randn(16), modality="smell")


def test_view_validation():
    with pytest.raises(ValueError):
        SkillBankView(torch.randn(3, 8), torch.randn(3, 16), torch.ones(3))  # float mask
    with pytest.raises(ValueError):
        SkillBankView(torch.randn(3, 8), torch.randn(4, 16), torch.ones(3, dtype=torch.bool))


def test_dual_memory_concatenates_tiers():
    memory = DualMemory(procedural_capacity=2, episodic_capacity=2, key_dim=8, value_dim=16)
    memory.write_event(torch.randn(8), torch.randn(16), metadata={"note": "hi"})
    memory.write_skill(torch.randn(8), torch.randn(16), modality="audio")
    assert len(memory) == 2
    view = memory.view()
    assert view.keys.shape == (2, 8) and view.values.shape == (2, 16)
    assert "hi" in memory.manifest()


def test_dual_memory_save_load_roundtrip(tmp_path):
    memory = DualMemory(procedural_capacity=2, episodic_capacity=2, key_dim=8, value_dim=16)
    key, value = torch.randn(8), torch.randn(16)
    memory.write_skill(key, value, modality="video", metadata={"clip": 7})
    memory.save(tmp_path / "mem.pt")
    restored = DualMemory(procedural_capacity=2, episodic_capacity=2, key_dim=8, value_dim=16)
    restored.load(tmp_path / "mem.pt")
    assert torch.equal(restored.procedural.values[0], value)
    assert restored.procedural.metadata[0]["clip"] == 7
    assert len(restored) == 1


def test_clear_resets_everything():
    bank = ProceduralSkillBank(capacity=2, key_dim=4, value_dim=4)
    bank.write(torch.ones(4), torch.ones(4))
    bank.clear()
    assert len(bank) == 0 and bank.view().empty
