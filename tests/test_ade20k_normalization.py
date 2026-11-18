import importlib
import importlib.util
from functools import partial
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import albumentations as A
import cv2
import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


ROOT = Path(__file__).resolve().parents[1]
MEAN = (0.48897059, 0.46548275, 0.4294)
STD = (0.22861765, 0.22948039, 0.24054667)
CUSTOM_MEAN = (0.11, 0.24, 0.37)
CUSTOM_STD = (0.42, 0.53, 0.64)


@pytest.fixture(scope="module")
def ade20k():
    # Load the actual dataset and sibling _common modules without running
    # mdistiller.dataset.__init__, which imports unrelated vision datasets.
    package = ModuleType("_singer_ade20k_test_dataset")
    package.__path__ = [str(ROOT / "mdistiller" / "dataset")]
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, package.__name__, package)
        yield importlib.import_module(f"{package.__name__}.ade20k")


@pytest.fixture
def data(ade20k, monkeypatch, tmp_path):
    for split, value, mask_value in [("training", 48, 2), ("validation", 160, 4)]:
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "annotations" / split).mkdir(parents=True)
        Image.new("RGB", (5, 4), (value,) * 3).save(tmp_path / "images" / split / "sample.jpg")
        Image.new("L", (5, 4), mask_value).save(tmp_path / "annotations" / split / "sample.png")
    (tmp_path / "objectInfo150.txt").write_text("1\tfixture\n", encoding="ascii")
    monkeypatch.setattr(ade20k, "ADE20k", partial(ade20k.ADE20k, dataroot=tmp_path))
    # The training-only wrapper uses an older Albumentations constructor.
    # Substitute its underlying jitter so this fixture tests loader wiring
    # and the real validation pipeline independently of that API mismatch.
    monkeypatch.setattr(ade20k, "SafeColorJitter", A.ColorJitter)
    common = sys.modules[f"{ade20k.__package__}._common"]
    # Use the actual distributed sampler without starting a process group.
    monkeypatch.setattr(common, "DistributedSampler", partial(DistributedSampler, num_replicas=2, rank=0))
    return ade20k


def statistics(loader):
    normalize, = (
        step for step in loader.dataset.transform.transforms if isinstance(step, A.Normalize)
    )
    return tuple(normalize.mean), tuple(normalize.std)


@pytest.mark.parametrize("mean,std,size,use_ddp,use_subset", [
    (None, None, 224, False, False),
    (CUSTOM_MEAN, CUSTOM_STD, 16, True, True),
], ids=["default", "custom"])
def test_shared_loader_uses_same_normalization_for_train_and_validation(
    data, monkeypatch, mean, std, size, use_ddp, use_subset
):
    forwarded = []
    original = data.get_ade20k_val_loader

    def capture(*args, **kwargs):
        forwarded.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(data, "get_ade20k_val_loader", capture)
    custom = {} if mean is None else {"mean": mean, "std": std}
    train, validation, _ = data.get_ade20k_dataloaders(
        3, 2, 0, use_ddp, use_subset=use_subset, img_size=size, **custom
    )
    expected = (MEAN, STD) if mean is None else (mean, std)
    assert statistics(train) == expected
    assert statistics(validation) == expected
    assert forwarded == [dict(use_subset=use_subset, mean=expected[0], std=expected[1], img_size=size)]


@pytest.mark.parametrize("use_ddp", [False, True], ids=["local", "distributed"])
def test_shared_loader_preserves_splits_transforms_and_loader_settings(data, use_ddp):
    train, validation, sample_count = data.get_ade20k_dataloaders(3, 2, 0, use_ddp, img_size=24)

    assert sample_count == len(train.dataset) == len(validation.dataset) == 1
    assert Path(train.dataset.image_root).name == Path(train.dataset.label_root).name == "training"
    assert Path(validation.dataset.image_root).name == Path(validation.dataset.label_root).name == "validation"
    assert (train.batch_size, train.num_workers, train.pin_memory) == (3, 0, True)
    assert (validation.batch_size, validation.num_workers, validation.pin_memory) == (2, 16, True)
    if use_ddp:
        assert isinstance(train.sampler, DistributedSampler) and train.sampler.shuffle
        assert isinstance(validation.sampler, DistributedSampler) and not validation.sampler.shuffle
    else:
        assert isinstance(train.sampler, RandomSampler)
        assert isinstance(validation.sampler, SequentialSampler)

    train_steps = train.dataset.transform.transforms
    assert [type(step).__name__ for step in train_steps] == [
        "RandomResizedCrop", "HorizontalFlip", "ColorJitter", "Normalize", "ToTensorV2",
    ]
    assert train_steps[0].size == (24, 24)
    val_steps = validation.dataset.transform.transforms
    assert [type(step).__name__ for step in val_steps] == ["SmallestMaxSize", "Resize", "Normalize", "ToTensorV2"]
    assert val_steps[0].max_size == 24
    assert (val_steps[1].height, val_steps[1].width, val_steps[1].interpolation) == (24, 24, cv2.INTER_LINEAR)
    image, label = validation.dataset[0]
    assert image.shape == (3, 24, 24)
    assert label.shape == (24, 24) and label.dtype == torch.int64
    assert torch.all(label == 3)


def test_direct_validation_loader_keeps_use_subset_compatibility(data):
    full = data.get_ade20k_val_loader(2, False, False, CUSTOM_MEAN, CUSTOM_STD, img_size=16)
    subset = data.get_ade20k_val_loader(2, False, True, CUSTOM_MEAN, CUSTOM_STD, img_size=16)

    assert full.dataset.filenames == subset.dataset.filenames
    assert statistics(full) == statistics(subset) == (CUSTOM_MEAN, CUSTOM_STD)
    full_image, full_label = full.dataset[0]
    subset_image, subset_label = subset.dataset[0]
    torch.testing.assert_close(full_image, subset_image)
    torch.testing.assert_close(full_label, subset_label)


def test_validation_pipeline_normalizes_real_pixels(data):
    _, validation, _ = data.get_ade20k_dataloaders(
        3, 2, 0, False, mean=CUSTOM_MEAN, std=CUSTOM_STD, img_size=16
    )
    image, label = validation.dataset[0]
    expected = torch.tensor((160 / 255 - np.array(CUSTOM_MEAN)) / np.array(CUSTOM_STD), dtype=torch.float32)
    torch.testing.assert_close(image, expected[:, None, None].expand(3, 16, 16))
    assert torch.all(label == 3)


@pytest.fixture
def evaluation_entry_points(data, monkeypatch):
    def placeholder(name, **members):
        module = ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, ROOT / path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    # Import both original scripts. Stand in for model, metric, checkpoint,
    # and distributed-training dependencies that the loader calls precede.
    placeholder("torchmetrics.classification", MulticlassJaccardIndex=None)
    placeholder("mdistiller.models._base", ModelBase=None)
    placeholder("mdistiller.models", imagenet_model_dict={})
    placeholder("mdistiller.utils", dist_fn=None)
    placeholder("tools.lineval.utils", init_parser=None, prepare_lineval_dir=None,
                load_from_checkpoint=None, load_head_checkpoint=None)
    monkeypatch.setitem(sys.modules, "mdistiller.dataset.ade20k", data)
    train = load("tools.lineval.ade20k", "tools/lineval/ade20k.py")
    standalone = load("tools.lineval.test.ade20k", "tools/lineval/test/ade20k.py")
    return train, standalone


@pytest.mark.parametrize("entry_index", [0, 1], ids=["training", "standalone"])
def test_evaluation_entry_points_use_correct_validation_normalization(
    data, evaluation_entry_points, monkeypatch, entry_index
):
    class ReachedLoaders(Exception):
        pass

    def stop_after_constructing_loaders(*args, **kwargs):
        _, val, _ = data.get_ade20k_dataloaders(*args, **kwargs)
        raise ReachedLoaders(val)

    entry = evaluation_entry_points[entry_index]
    monkeypatch.setattr(entry, "get_ade20k_dataloaders", stop_after_constructing_loaders)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("IS_MASTER_NODE", "0")
    monkeypatch.setattr(entry, "world_size", 2, raising=False)
    args = SimpleNamespace(epochs=1, batch_size=4, test_batch_size=6, num_workers=0, img_size=32, device="cpu")
    with pytest.raises(ReachedLoaders) as status:
        entry.main(args)

    val = status.value.args[0]
    assert val.batch_size == (3 if entry_index == 0 else 4)
    assert isinstance(val.sampler, DistributedSampler) == (entry_index == 0)
    assert val.dataset.transform.transforms[1].height == 32
    assert statistics(val) == (MEAN, STD)
