import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Flower Species Recognition — Oxford 102 · FlowerCNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python main.py --mode download             # fetch Oxford 102 dataset
  python main.py --mode train                # train from scratch (MPS/CUDA/CPU)
  python main.py --mode evaluate             # evaluate on held-out test split
  python main.py --mode app                  # launch Flask demo at :5000
  python main.py --mode train --mode evaluate  # train then immediately evaluate
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["download", "train", "evaluate", "app"],
        action="append",           # allows multiple --mode flags
        dest="modes",
        required=True,
        metavar="MODE",
        help="Operation to run: download | train | evaluate | app",
    )
    return parser


def runDownload():
    print("\n━━━  Mode: download  ━━━")
    from downloadData import downloadOxford102
    import config
    downloadOxford102(config.RAW_DIR)


def runTrain():
    print("\n━━━  Mode: train  ━━━")
    from train.trainer import runTraining
    runTraining()


def runEvaluate():
    print("\n━━━  Mode: evaluate  ━━━")
    from evaluate.evaluator import runEvaluation
    runEvaluation()


def runApp():
    print("\n━━━  Mode: app  ━━━")
    from app.flaskApp import runApp as startFlask
    startFlask()


_HANDLERS = {
    "download": runDownload,
    "train":    runTrain,
    "evaluate": runEvaluate,
    "app":      runApp,
}


def checkDatasetReady() -> bool:
    import config
    return config.ORGANIZED_DIR.exists() and any(config.ORGANIZED_DIR.iterdir())


def checkCheckpointReady() -> bool:
    import config
    return config.CNN_CHECKPOINT.exists() and config.LABEL_MAP_CACHE.exists()


def validateModePrerequisites(modes: list[str]):

    import config

    if "train" in modes and not checkDatasetReady():
        print(
            "\n  ✗ Dataset not found.\n"
            f"    Expected: {config.ORGANIZED_DIR}\n"
            "    Run:  python main.py --mode download\n"
        )
        sys.exit(1)

    if "evaluate" in modes and "train" not in modes and not checkCheckpointReady():
        print(
            "\n  ✗ No trained model found.\n"
            f"    Expected: {config.CNN_CHECKPOINT}\n"
            "    Run:  python main.py --mode train\n"
        )
        sys.exit(1)

    if "app" in modes and "train" not in modes and not checkCheckpointReady():
        print(
            "\n  ✗ No trained model found — cannot start the app.\n"
            f"    Expected: {config.CNN_CHECKPOINT}\n"
            "    Run:  python main.py --mode train\n"
        )
        sys.exit(1)


def main():
    parser = buildParser()
    args   = parser.parse_args()
    modes  = args.modes
    seen  = set()
    modes = [m for m in modes if not (m in seen or seen.add(m))]

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║   Flower Species Recognition · Oxford 102 · FlowerCNN       ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"  Modes     : {' → '.join(modes)}")

    import config
    print(f"  Device    : {config.DEVICE}"
          f"{'  ← Apple Silicon MPS ✓' if str(config.DEVICE) == 'mps' else ''}")
    print(f"  Root      : {ROOT}")
    print("╚══════════════════════════════════════════════════════════════╝")

    validateModePrerequisites(modes)

    for mode in modes:
        _HANDLERS[mode]()


if __name__ == "__main__":
    main()