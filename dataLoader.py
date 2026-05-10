import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Tuple, Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image

import config
from augment.transforms import getTrainTransforms, getInferenceTransforms


class FlowerDataset(Dataset):

    VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")

    def __init__(
        self,
        organizedDir: Path,
        transform=None,
    ):
        super().__init__()

        self.organizedDir = Path(organizedDir)
        self.transform    = transform

        if not self.organizedDir.exists():
            raise FileNotFoundError(
                f"Organized dataset directory not found: {self.organizedDir}\n"
                f"Run 'python downloadData.py' first to download and organise the dataset."
            )

        self.classes    = sorted([
            d.name for d in self.organizedDir.iterdir()
            if d.is_dir() and d.name.startswith("class_")
        ])

        if len(self.classes) == 0:
            raise RuntimeError(
                f"No class sub-folders found in {self.organizedDir}.\n"
                f"Expected folders named 'class_001_…', 'class_002_…', etc."
            )

        self.classToIdx: Dict[str, int] = {
            cls: idx for idx, cls in enumerate(self.classes)
        }

        self.idxToName: Dict[int, str] = {
            idx: cls.split("_", 2)[-1]          # e.g. "pinkPrimrose"
            for cls, idx in self.classToIdx.items()
        }

        self.samples: List[Tuple[Path, int]] = []
        for cls in self.classes:
            classDir = self.organizedDir / cls
            label    = self.classToIdx[cls]
            for imgPath in sorted(classDir.iterdir()):
                if imgPath.suffix.lower() in self.VALID_EXTENSIONS:
                    self.samples.append((imgPath, label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found under {self.organizedDir}. "
                f"Check that the download and organisation completed successfully."
            )


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        imgPath, label = self.samples[idx]

        image = Image.open(imgPath)
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


    def getClassWeights(self) -> torch.Tensor:

        classCounts = defaultdict(int)
        for _, label in self.samples:
            classCounts[label] += 1

        numClasses = len(self.classes)
        totalSamples = len(self.samples)
        weights = torch.zeros(numClasses, dtype=torch.float32)
        for classIdx, count in classCounts.items():
            # Inverse frequency: rare classes get higher weight
            weights[classIdx] = totalSamples / (numClasses * count)

        return weights


def stratifiedSplit(
    dataset: FlowerDataset,
    trainRatio: float = config.TRAIN_SPLIT,
    valRatio:   float = config.VAL_SPLIT,
    seed:       int   = 42,
) -> Tuple[List[int], List[int], List[int]]:

    rng = random.Random(seed)

    classIndices: Dict[int, List[int]] = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        classIndices[label].append(idx)

    trainIdx, valIdx, testIdx = [], [], []

    for label in sorted(classIndices.keys()):
        indices = classIndices[label]
        rng.shuffle(indices)

        n         = len(indices)
        nTrain    = max(1, int(n * trainRatio))
        nVal      = max(1, int(n * valRatio))
        nTest     = max(1, n - nTrain - nVal)

        if nTrain + nVal + nTest > n:
            nTest = n - nTrain - nVal

        trainIdx.extend(indices[:nTrain])
        valIdx.extend(  indices[nTrain : nTrain + nVal])
        testIdx.extend( indices[nTrain + nVal : nTrain + nVal + nTest])

    return trainIdx, valIdx, testIdx

def getDataLoaders(
    organizedDir: Path  = config.ORGANIZED_DIR,
    batchSize:    int   = config.BATCH_SIZE,
    numWorkers:   int   = config.NUM_WORKERS,
    pinMemory:    bool  = config.PIN_MEMORY,
    seed:         int   = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str]]:

    baseDataset = FlowerDataset(organizedDir, transform=None)

    trainIdx, valIdx, testIdx = stratifiedSplit(baseDataset, seed=seed)


    trainDataset = FlowerDataset(organizedDir, transform=getTrainTransforms())
    valDataset   = FlowerDataset(organizedDir, transform=getInferenceTransforms())
    testDataset  = FlowerDataset(organizedDir, transform=getInferenceTransforms())

    trainSubset = Subset(trainDataset, trainIdx)
    valSubset   = Subset(valDataset,   valIdx)
    testSubset  = Subset(testDataset,  testIdx)

    trainLoader = DataLoader(
        trainSubset,
        batch_size=batchSize,
        shuffle=True,
        num_workers=numWorkers,
        pin_memory=pinMemory,
        drop_last=True,
    )

    valLoader = DataLoader(
        valSubset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=numWorkers,
        pin_memory=pinMemory,
        drop_last=False,
    )

    testLoader = DataLoader(
        testSubset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=numWorkers,
        pin_memory=pinMemory,
        drop_last=False,
    )

    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.LABEL_MAP_CACHE, "w") as f:
        # JSON keys must be strings — convert int keys to str
        json.dump(
            {str(k): v for k, v in baseDataset.idxToName.items()},
            f, indent=2
        )

    return trainLoader, valLoader, testLoader, baseDataset.idxToName

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    print("\n╔══════════════════════════════════════════════╗")
    print("║   utils/dataLoader.py — Sanity Check         ║")
    print("╚══════════════════════════════════════════════╝")

    organizedDir = config.ORGANIZED_DIR

    if not organizedDir.exists():
        print(f"\n  ✗ Organized dataset not found at:\n    {organizedDir}")
        print("  Run 'python downloadData.py' first.\n")
        sys.exit(1)

    print(f"  Loading dataset from: {organizedDir}")
    trainLoader, valLoader, testLoader, idxToName = getDataLoaders()

    print(f"\n  Classes found       : {len(idxToName)}")
    print(f"  Train batches       : {len(trainLoader)}")
    print(f"  Val   batches       : {len(valLoader)}")
    print(f"  Test  batches       : {len(testLoader)}")
    print(f"  Train samples       : {len(trainLoader.dataset)}")
    print(f"  Val   samples       : {len(valLoader.dataset)}")
    print(f"  Test  samples       : {len(testLoader.dataset)}")
    print(f"  Label map cached at : {config.LABEL_MAP_CACHE}")

    images, labels = next(iter(trainLoader))
    print(f"\n  Sample batch shape  : {tuple(images.shape)}  ← (B, C, H, W)")
    print(f"  Sample label shape  : {tuple(labels.shape)}")
    print(f"  Label range         : [{labels.min().item()}, {labels.max().item()}]")
    print(f"  Pixel value range   : [{images.min():.3f}, {images.max():.3f}]")

    print(f"\n  First 5 class names:")
    for i in range(min(5, len(idxToName))):
        print(f"    [{i:>3}] {idxToName[i]}")

    assert images.shape[1:] == (3, 224, 224), "Image tensor shape mismatch"
    assert labels.min() >= 0 and labels.max() < 102, "Label index out of range"
    print("\n  ✓ All assertions passed\n")