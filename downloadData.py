import os
import sys
import argparse
import tarfile
import shutil
import json
import urllib.request
from pathlib import Path

import scipy.io

OXFORD102_URL        = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
OXFORD102_LABELS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
OXFORD102_SPLITS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"


OXFORD102_CLASSES = [
    "pinkPrimrose", "hardLeavedPocketOrchid", "canterburyBells", "sweetPea",
    "englishMarigold", "tigerLily", "moon Orchid", "bird OfParadise", "monkshood",
    "globe Thistle", "snapdragon", "coltsFoot", "kingProtea", "spear Thistle",
    "yellowIris", "globeFlower", "purpleConeflower", "peruvianLily", "balloon Flower",
    "giantWhiteArumLily", "fireAndIce", "pincushionFlower", "fritillary",
    "redGinger", "gropebLily", "princeOfWales Feathers", "stemlessGentian",
    "artichoke", "sweetWilliam", "carnation", "gardenPhlox", "loveInTheM ist",
    "mexicanAster", "alpineSeaCampion", "rubySpikedCranesbill", "watercress",
    "wall flower", "lotuslotus", "gardenRoseRosaceae", "purplePetunia",
    "bubBluebell", "windFlower", "pelargonium", "prince OfWalesFeathers",
    "osteospermum", "clematisLilacViolet", "anthurium", "frangipani",
    "clematis", "hibiscus", "columbine", "desertRose", "treeMallow",
    "magnolia", "cyclamen", "waterlily", "roseRosaceae", "monkeyface Orchid",
    "azalea", "whiteOrchid", "passionFlower", "lotus", "thorn Apple",
    "bird-Of-Paradise", "camellia", "mallowMalvaceae", "mexicanPetunia",
    "bougainvillea", "valeriana", "snapdragonAntirrhinum", "hemerocallis",
    "morning Glory", "passion Flower", "pinkYellow Dahlia", "prince-Of-Wales-Feathers",
    "primula", "sunflower", "pelargoniumGeranium", "bishop-Of-Llandaff",
    "daisy", "cow Parsley", "siam Tulip", "love In The Mist", "mexican Aster",
    "alpine Sea Campion", "stemless Gentian", "silene", "californiaPoppy",
    "canterbury Bells", "rose", "petunia", "wildPansy", "primulaVulgaris",
    "pinkDaisy", "osteospermumAfrica", "springCrocus", "bearded Iris",
    "windmillPalm", "monkshood2", "bromelia", "blanket Flower",
    "trumpet Creeper", "blackberry Lily"
]


def showProgress(blockNum, blockSize, totalSize):
    downloaded = blockNum * blockSize
    percent = min(100.0, downloaded * 100.0 / totalSize) if totalSize > 0 else 0
    bar = int(percent / 2)
    sys.stdout.write(
        f"\r  [{'█' * bar}{'░' * (50 - bar)}] {percent:5.1f}%  "
        f"({downloaded / 1e6:.1f} / {totalSize / 1e6:.1f} MB)"
    )
    sys.stdout.flush()
    if downloaded >= totalSize:
        print()


def downloadFile(url: str, dest: Path, label: str):
    if dest.exists():
        print(f"  ✓ Already downloaded: {dest.name}")
        return
    print(f"  ↓ Downloading {label} …")
    urllib.request.urlretrieve(url, dest, showProgress)
    print(f"  ✓ Saved → {dest}")


def extractTar(tarPath: Path, extractTo: Path):
    print(f"  ⟳ Extracting {tarPath.name} …")
    with tarfile.open(tarPath, "r:gz") as t:
        t.extractall(extractTo)
    print(f"  ✓ Extracted → {extractTo}")


def writeClassMap(classMap: dict, dest: Path):
    with open(dest, "w") as f:
        json.dump(classMap, f, indent=2)
    print(f"  ✓ Class map written → {dest}")


def downloadOxford102(rawDir: Path):
    print("\n━━━  Oxford 102-Flower Dataset  ━━━")
    targetDir = rawDir / "oxford102"
    targetDir.mkdir(parents=True, exist_ok=True)

    tgzPath = targetDir / "102flowers.tgz"
    downloadFile(OXFORD102_URL, tgzPath, "Oxford 102 images")

    labelsPath = targetDir / "imagelabels.mat"
    splitsPath = targetDir / "setid.mat"
    downloadFile(OXFORD102_LABELS_URL, labelsPath, "image labels (.mat)")
    downloadFile(OXFORD102_SPLITS_URL, splitsPath, "train/val/test splits (.mat)")

    jpgDir = targetDir / "jpg"
    if not jpgDir.exists():
        extractTar(tgzPath, targetDir)

    organizedDir = targetDir / "organized"
    if organizedDir.exists():
        print("  ✓ Already organized — skipping reorganisation")
    else:
        organizedDir.mkdir()

        print("  Loading imagelabels.mat …")
        labelsData = scipy.io.loadmat(str(labelsPath))
        labels = labelsData["labels"].flatten()

        allImages = sorted(jpgDir.glob("image_*.jpg"))
        print(f"  Found {len(allImages)} images — organising into 102 class folders …")

        for imgPath in allImages:
            imgNumber = int(imgPath.stem.split("_")[1]) - 1
            if imgNumber >= len(labels):
                continue
            classIdx = int(labels[imgNumber]) - 1
            if classIdx >= len(OXFORD102_CLASSES):
                continue
            className = OXFORD102_CLASSES[classIdx]
            safeClassName = className.replace(" ", "").replace("-", "")
            classDir = organizedDir / f"class_{classIdx+1:03d}_{safeClassName}"
            classDir.mkdir(exist_ok=True)
            shutil.copy2(imgPath, classDir / imgPath.name)

        print("  ✓ Reorganised into 102 class sub-folders")

    classMap = {
        f"class_{i+1:03d}": {
            "index": i,
            "name": OXFORD102_CLASSES[i],
            "label": OXFORD102_CLASSES[i].lower()
        }
        for i in range(102)
    }
    writeClassMap(classMap, targetDir / "classMap.json")
    print(f"  ✓ Oxford 102 ready at: {targetDir}")
    return targetDir / "organized"

def main():
    parser = argparse.ArgumentParser(
        description="Download the Oxford 102-Flower dataset from robots.ox.ac.uk"
    )
    parser.add_argument(
        "--dataDir",
        default="data/raw",
        help="Root directory to store raw downloads (default: data/raw)"
    )
    args = parser.parse_args()

    rawDir = Path(args.dataDir)
    rawDir.mkdir(parents=True, exist_ok=True)

    print("\n╔══════════════════════════════════════════════╗")
    print("║   Oxford 102-Flower Dataset Downloader       ║")
    print("║   Flower Species Recognition Project         ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  Output root: {rawDir.resolve()}\n")

    try:
        import scipy.io
    except ImportError:
        print("  ✗ scipy not found. Installing …")
        os.system(f"{sys.executable} -m pip install scipy --quiet")
        import scipy.io

    downloadOxford102(rawDir)

    print("\n╔══════════════════════════════════════════════╗")
    print("║   Download complete ✓                        ║")
    print("╚══════════════════════════════════════════════╝\n")
    print("  Next step:")
    print("    python main.py --mode train")
    print("    python main.py --mode app\n")


if __name__ == "__main__":
    main()