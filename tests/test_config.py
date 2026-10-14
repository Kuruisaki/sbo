from pathlib import Path
import shutil
from uuid import uuid4

import pytest
import yaml

from sbo.config import ConfigurationError, FrozenConfig, create_run, load_config, verify_run_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def data():
    return yaml.safe_load((ROOT / "configs/naca2412.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("mode", ["unknown", "missing", "boolean_number", "empty_unknown"])
def test_strict_schema(data, mode):
    if mode == "unknown": data["flow"]["typo"] = 1
    if mode == "missing": del data["flow"]["mach"]
    if mode == "boolean_number": data["flow"]["mach"] = True
    if mode == "empty_unknown": data["unexpected"] = {}
    with pytest.raises(ConfigurationError):
        FrozenConfig(data)


def test_freeze_is_immutable_and_copy_is_independent(data):
    frozen = FrozenConfig(data)
    data["flow"]["mach"] = .2
    assert frozen["flow"]["mach"] == .1
    with pytest.raises(TypeError):
        frozen["flow"]["mach"] = .2
    assert isinstance(frozen["design"]["bounds"], tuple)
    other = frozen.to_dict()
    other["flow"]["mach"] = .2
    assert frozen["flow"]["mach"] == .1


def test_duplicate_keys_and_existing_run_are_rejected(data):
    root = ROOT / "runs" / f"config-tests-{uuid4().hex}"
    root.mkdir()
    try:
        duplicate = root / "duplicate.yaml"
        duplicate.write_text("flow: {mach: 0.1, mach: 0.2}\n")
        with pytest.raises(ConfigurationError, match="duplicate"):
            load_config(duplicate)
        data["runtime"]["runs_root"] = str(root)
        config = FrozenConfig(data)
        directory = create_run(config, "run_1")
        verify_run_config(directory, config)
        with pytest.raises(FileExistsError):
            create_run(config, "run_1")
        with pytest.raises(ConfigurationError):
            create_run(config, "../escape")
    finally:
        assert root.resolve().parent == (ROOT / "runs").resolve()
        shutil.rmtree(root)
