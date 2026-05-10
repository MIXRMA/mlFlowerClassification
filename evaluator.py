import json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")        # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)
from tqdm import tqdm

import config
from models.cnn import FlowerCNN
from utils.dataLoader import getDataLoaders, FlowerDataset

class Evaluator:

    def __init__(
        self,
        checkpointPath: Path = None,
        device: torch.device = None,
    ):
        self.checkpointPath = checkpointPath or config.CNN_CHECKPOINT
        self.device         = device or config.DEVICE
        self.model          = None
        self.idxToName      = None

    def _loadModel(self) -> FlowerCNN:
    
        if not self.checkpointPath.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpointPath}\n"
                f"Run 'python main.py --mode train' first."
            )

        checkpoint = torch.load(
            self.checkpointPath,
            map_location=self.device,
            weights_only=False,
        )

        model = FlowerCNN(
            numClasses=checkpoint.get("numClasses",   config.NUM_CLASSES),
            convChannels=checkpoint.get("convChannels", config.CONV_CHANNELS),
            fcHiddenDim=checkpoint.get("fcHiddenDim",  config.FC_HIDDEN_DIM),
            dropoutRate=checkpoint.get("dropoutRate",   config.DROPOUT_RATE),
        ).to(self.device)

        model.load_state_dict(checkpoint["modelStateDict"])
        model.eval()

        print(f"  ✓ Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} "
              f"(val acc: {checkpoint.get('valAcc', 0)*100:.2f}%)")
        return model

    def _loadLabelMap(self) -> dict:
        """Loads idx→className mapping from cached JSON."""
        if not config.LABEL_MAP_CACHE.exists():
            raise FileNotFoundError(
                f"Label map not found: {config.LABEL_MAP_CACHE}\n"
                f"Run 'python main.py --mode train' first to generate it."
            )
        with open(config.LABEL_MAP_CACHE) as f:
            rawMap = json.load(f)
        # JSON keys are strings — convert back to int
        return {int(k): v for k, v in rawMap.items()}

    @torch.no_grad()
    def _collectPredictions(
        self,
        testLoader,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        self.model.eval()
        labelsList = []
        predsList  = []
        probsList  = []

        pbar = tqdm(testLoader, desc="  Evaluating", ncols=80, unit="batch")

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.model(images)
            probs  = F.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)

            labelsList.append(labels.cpu().numpy())
            predsList.append(preds.cpu().numpy())
            probsList.append(probs.cpu().numpy())

        allLabels = np.concatenate(labelsList)
        allPreds  = np.concatenate(predsList)
        allProbs  = np.concatenate(probsList)

        return allLabels, allPreds, allProbs

    def _computeMetrics(
        self,
        allLabels: np.ndarray,
        allPreds:  np.ndarray,
        allProbs:  np.ndarray,
    ) -> dict:
   
        numClasses = allProbs.shape[1]

        top1Acc = float((allPreds == allLabels).mean())

        top5Acc = top_k_accuracy_score(
            allLabels, allProbs,
            k=min(5, numClasses),
            labels=list(range(numClasses)),
        )

        targetNames = [self.idxToName.get(i, str(i)) for i in range(numClasses)]
        report = classification_report(
            allLabels, allPreds,
            target_names=targetNames,
            zero_division=0,
            output_dict=True,
        )

        weightedF1 = report["weighted avg"]["f1-score"]
        macroF1    = report["macro avg"]["f1-score"]

        perClassAcc = {}
        for classIdx in range(numClasses):
            mask         = allLabels == classIdx
            if mask.sum() == 0:
                continue
            classAcc     = float((allPreds[mask] == allLabels[mask]).mean())
            className    = self.idxToName.get(classIdx, str(classIdx))
            perClassAcc[className] = round(classAcc, 4)

        return {
            "top1Accuracy": round(top1Acc,    4),
            "top5Accuracy": round(top5Acc,    4),
            "weightedF1":   round(weightedF1, 4),
            "macroF1":      round(macroF1,    4),
            "perClassAcc":  perClassAcc,
            "classReport":  report,
        }


    def _plotConfusionMatrix(
        self,
        allLabels:  np.ndarray,
        allPreds:   np.ndarray,
        outputPath: Path,
        maxClasses: int = 30,
    ):
      
        numClasses = int(allLabels.max()) + 1

        perClassAcc = []
        for i in range(numClasses):
            mask = allLabels == i
            if mask.sum() > 0:
                acc = float((allPreds[mask] == i).mean())
                perClassAcc.append((i, acc))

        perClassAcc.sort(key=lambda x: x[1])
        selectedIndices = [idx for idx, _ in perClassAcc[:maxClasses]]
        selectedIndices.sort()
        mask         = np.isin(allLabels, selectedIndices)
        filteredTrue = allLabels[mask]
        filteredPred = allPreds[mask]

        idxRemap  = {oldIdx: newIdx for newIdx, oldIdx in enumerate(selectedIndices)}
        remappedTrue = np.array([idxRemap[i] for i in filteredTrue])
        remappedPred = np.array([idxRemap.get(i, -1) for i in filteredPred])

        validMask    = remappedPred >= 0
        remappedTrue = remappedTrue[validMask]
        remappedPred = remappedPred[validMask]

        cm          = confusion_matrix(remappedTrue, remappedPred)
        cmNorm      = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

        classLabels = [self.idxToName.get(i, str(i)) for i in selectedIndices]

        figSize = max(12, len(selectedIndices) * 0.4)
        fig, ax = plt.subplots(figsize=(figSize, figSize * 0.85))

        sns.heatmap(
            cmNorm,
            ax=ax,
            xticklabels=classLabels,
            yticklabels=classLabels,
            cmap="YlOrRd",
            vmin=0.0, vmax=1.0,
            linewidths=0.3,
            linecolor="gray",
            annot=len(selectedIndices) <= 20,   # only annotate if not too dense
            fmt=".2f",
            cbar_kws={"label": "Recall (row-normalised)"},
        )

        ax.set_title(
            f"Confusion Matrix — Top {len(selectedIndices)} Most Confused Classes\n"
            f"FlowerCNN · Oxford 102",
            fontsize=13, pad=14,
        )
        ax.set_xlabel("Predicted Class", fontsize=10)
        ax.set_ylabel("True Class",      fontsize=10)
        ax.tick_params(axis="x", rotation=90,  labelsize=7)
        ax.tick_params(axis="y", rotation=0,   labelsize=7)

        plt.tight_layout()
        outputPath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outputPath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Confusion matrix saved → {outputPath}")


    def evaluate(self) -> dict:
        """
        Full evaluation pipeline:
          1. Load model and label map
          2. Build test DataLoader
          3. Collect predictions
          4. Compute and print metrics
          5. Save confusion matrix PNG
          6. Write metrics JSON

        Returns:
            metrics dict with top1Accuracy, top5Accuracy, weightedF1, etc.
        """
        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║   FlowerCNN — Evaluation                                     ║")
        print("╠══════════════════════════════════════════════════════════════╣")
        print(f"  Checkpoint : {self.checkpointPath}")
        print(f"  Device     : {self.device}")
        print("╚══════════════════════════════════════════════════════════════╝\n")

        self.model     = self._loadModel()
        self.idxToName = self._loadLabelMap()

        print("  Building test DataLoader …")
        _, _, testLoader, _ = getDataLoaders()

        allLabels, allPreds, allProbs = self._collectPredictions(testLoader)

        metrics = self._computeMetrics(allLabels, allPreds, allProbs)

        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║   Evaluation Results                                         ║")
        print("╠══════════════════════════════════════════════════════════════╣")
        print(f"  Top-1 Accuracy : {metrics['top1Accuracy']*100:.2f}%")
        print(f"  Top-5 Accuracy : {metrics['top5Accuracy']*100:.2f}%")
        print(f"  Weighted F1    : {metrics['weightedF1']:.4f}")
        print(f"  Macro F1       : {metrics['macroF1']:.4f}")
        print(f"  Test samples   : {len(allLabels)}")
        print("╠══════════════════════════════════════════════════════════════╣")

        sortedClasses = sorted(
            metrics["perClassAcc"].items(), key=lambda x: x[1]
        )
        print("  Worst 5 classes:")
        for name, acc in sortedClasses[:5]:
            print(f"    {name:<35} {acc*100:5.1f}%")
        print("  Best 5 classes:")
        for name, acc in sortedClasses[-5:]:
            print(f"    {name:<35} {acc*100:5.1f}%")
        print("╚══════════════════════════════════════════════════════════════╝\n")

        cmPath = config.CHECKPOINTS_DIR / "confusionMatrix.png"
        self._plotConfusionMatrix(allLabels, allPreds, cmPath)

        metricsPath = config.CHECKPOINTS_DIR / "evalMetrics.json"
        saveMetrics = {k: v for k, v in metrics.items() if k != "classReport"}
        with open(metricsPath, "w") as f:
            json.dump(saveMetrics, f, indent=2)
        print(f"  ✓ Metrics saved → {metricsPath}\n")

        return metrics


def runEvaluation() -> dict:
    """Called by main.py --mode evaluate."""
    evaluator = Evaluator()
    return evaluator.evaluate()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    print("\n╔══════════════════════════════════════════════╗")
    print("║   evaluate/evaluator.py — Sanity Check       ║")
    print("║   (synthetic data, no checkpoint needed)     ║")
    print("╚══════════════════════════════════════════════╝\n")

    device     = config.getDevice()
    numClasses = 10
    nSamples   = 200

    np.random.seed(42)
    syntheticLabels = np.random.randint(0, numClasses, nSamples)
    syntheticPreds  = syntheticLabels.copy()
    noiseIdx        = np.random.choice(nSamples, size=int(nSamples * 0.4), replace=False)
    syntheticPreds[noiseIdx] = np.random.randint(0, numClasses, len(noiseIdx))

    syntheticProbs  = np.random.dirichlet(np.ones(numClasses), size=nSamples)
    for i, trueClass in enumerate(syntheticLabels):
        syntheticProbs[i, trueClass] += 1.5
    syntheticProbs /= syntheticProbs.sum(axis=1, keepdims=True)

    syntheticIdxToName = {i: f"flower_{i:02d}" for i in range(numClasses)}

    evaluator            = Evaluator()
    evaluator.model      = None      # not needed for metric computation check
    evaluator.idxToName  = syntheticIdxToName

    metrics = evaluator._computeMetrics(
        syntheticLabels, syntheticPreds, syntheticProbs
    )

    print(f"  Top-1 Accuracy : {metrics['top1Accuracy']*100:.2f}%  (expect ~60%)")
    print(f"  Top-5 Accuracy : {metrics['top5Accuracy']*100:.2f}%")
    print(f"  Weighted F1    : {metrics['weightedF1']:.4f}")
    print(f"  Macro F1       : {metrics['macroF1']:.4f}")
    print(f"  Per-class keys : {list(metrics['perClassAcc'].keys())[:3]} …")

    cmPath = config.CHECKPOINTS_DIR / "confusionMatrix_smoketest.png"
    evaluator._plotConfusionMatrix(syntheticLabels, syntheticPreds, cmPath, maxClasses=10)

    assert 0.50 <= metrics["top1Accuracy"] <= 0.75, "Top-1 accuracy out of expected range"
    assert metrics["top5Accuracy"] >= metrics["top1Accuracy"], "Top-5 must be ≥ Top-1"
    assert cmPath.exists(), "Confusion matrix PNG not created"

    print("\n  ✓ All assertions passed")
    print(f"  ✓ Confusion matrix smoke test saved → {cmPath}\n")