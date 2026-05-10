import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from models.cnn import FlowerCNN
from utils.dataLoader import FlowerDataset, getDataLoaders


class Trainer:

    def __init__(
        self,
        device:       torch.device = None,
        numClasses:   int          = config.NUM_CLASSES,
        learningRate: float        = config.LEARNING_RATE,
        weightDecay:  float        = config.WEIGHT_DECAY,
        numEpochs:    int          = config.NUM_EPOCHS,
        patience:     int          = config.EARLY_STOP_PATIENCE,
    ):
        self.device       = device or config.DEVICE
        self.numClasses   = numClasses
        self.learningRate = learningRate
        self.weightDecay  = weightDecay
        self.numEpochs    = numEpochs
        self.patience     = patience

        self.model      = None
        self.optimiser  = None
        self.scheduler  = None
        self.criterion  = None

        self.history = {
            "trainLoss": [], "trainAcc": [],
            "valLoss":   [], "valAcc":   [],
        }


    def _buildModel(self) -> FlowerCNN:
        model = FlowerCNN(numClasses=self.numClasses).to(self.device)
        return model

    def _buildCriterion(self, classWeights: torch.Tensor) -> nn.CrossEntropyLoss:
       
        return nn.CrossEntropyLoss(
            weight=classWeights.to(self.device)
        )

    def _buildOptimiser(self) -> torch.optim.Adam:
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.learningRate,
            weight_decay=self.weightDecay,
        )

    def _buildScheduler(self) -> torch.optim.lr_scheduler.StepLR:
    
        return torch.optim.lr_scheduler.StepLR(
            self.optimiser,
            step_size=config.LR_STEP_SIZE,
            gamma=config.LR_GAMMA,
        )


    def _trainEpoch(self, trainLoader: DataLoader) -> tuple[float, float]:
       
        self.model.train()
        totalLoss    = 0.0
        totalCorrect = 0
        totalSamples = 0

        pbar = tqdm(
            trainLoader,
            desc="  Train",
            leave=False,
            ncols=80,
            unit="batch",
        )

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimiser.zero_grad()
            logits = self.model(images)
            loss   = self.criterion(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimiser.step()

            batchSize     = images.size(0)
            totalLoss    += loss.item() * batchSize
            predictions   = logits.argmax(dim=1)
            totalCorrect += (predictions == labels).sum().item()
            totalSamples += batchSize

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avgLoss  = totalLoss / totalSamples
        accuracy = totalCorrect / totalSamples
        return avgLoss, accuracy

    @torch.no_grad()
    def _valEpoch(self, valLoader: DataLoader) -> tuple[float, float]:

        self.model.eval()
        totalLoss    = 0.0
        totalCorrect = 0
        totalSamples = 0

        pbar = tqdm(
            valLoader,
            desc="  Val  ",
            leave=False,
            ncols=80,
            unit="batch",
        )

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss   = self.criterion(logits, labels)

            batchSize     = images.size(0)
            totalLoss    += loss.item() * batchSize
            predictions   = logits.argmax(dim=1)
            totalCorrect += (predictions == labels).sum().item()
            totalSamples += batchSize

        avgLoss  = totalLoss / totalSamples
        accuracy = totalCorrect / totalSamples
        return avgLoss, accuracy


    def train(
        self,
        trainLoader:  DataLoader,
        valLoader:    DataLoader,
        classWeights: torch.Tensor,
    ) -> FlowerCNN:
   
        self.model     = self._buildModel()
        self.criterion = self._buildCriterion(classWeights)
        self.optimiser = self._buildOptimiser()
        self.scheduler = self._buildScheduler()

        bestValAcc    = 0.0
        bestEpoch     = 0
        epochsNoImprove = 0

        totalParams = self.model.countParameters()

        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║   FlowerCNN — Training                                       ║")
        print("╠══════════════════════════════════════════════════════════════╣")
        print(f"  Device          : {self.device}")
        print(f"  Trainable params: {totalParams:,}")
        print(f"  Epochs (max)    : {self.numEpochs}")
        print(f"  Batch size      : {config.BATCH_SIZE}")
        print(f"  Learning rate   : {self.learningRate}")
        print(f"  Early stop      : {self.patience} epochs patience")
        print(f"  Checkpoint      : {config.CNN_CHECKPOINT}")
        print("╚══════════════════════════════════════════════════════════════╝\n")

        trainingStartTime = time.time()

        for epoch in range(1, self.numEpochs + 1):
            epochStart = time.time()
            currentLr  = self.optimiser.param_groups[0]["lr"]

            print(f"Epoch [{epoch:>3}/{self.numEpochs}]  LR: {currentLr:.2e}")

            trainLoss, trainAcc = self._trainEpoch(trainLoader)

            valLoss, valAcc = self._valEpoch(valLoader)

            self.scheduler.step()

            self.history["trainLoss"].append(round(trainLoss, 6))
            self.history["trainAcc"].append( round(trainAcc,  6))
            self.history["valLoss"].append(  round(valLoss,   6))
            self.history["valAcc"].append(   round(valAcc,    6))

            epochTime = time.time() - epochStart

            improved = "  ← best ✓" if valAcc > bestValAcc else ""
            print(
                f"  train loss: {trainLoss:.4f}  train acc: {trainAcc*100:5.2f}%"
                f"  |  val loss: {valLoss:.4f}  val acc: {valAcc*100:5.2f}%"
                f"  [{epochTime:.1f}s]{improved}"
            )

            if valAcc > bestValAcc:
                bestValAcc       = valAcc
                bestEpoch        = epoch
                epochsNoImprove  = 0

                config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch":          epoch,
                        "modelStateDict": self.model.state_dict(),
                        "optimiserState": self.optimiser.state_dict(),
                        "valAcc":         valAcc,
                        "valLoss":        valLoss,
                        "numClasses":     self.numClasses,
                        "convChannels":   config.CONV_CHANNELS,
                        "fcHiddenDim":    config.FC_HIDDEN_DIM,
                        "dropoutRate":    config.DROPOUT_RATE,
                    },
                    config.CNN_CHECKPOINT,
                )
            else:
                epochsNoImprove += 1

            if epochsNoImprove >= self.patience:
                print(f"\n  Early stopping triggered — no improvement for "
                      f"{self.patience} consecutive epochs.")
                print(f"  Best val acc: {bestValAcc*100:.2f}% at epoch {bestEpoch}\n")
                break

        totalTime = time.time() - trainingStartTime
        minutes, seconds = divmod(int(totalTime), 60)

        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║   Training Complete                                          ║")
        print("╠══════════════════════════════════════════════════════════════╣")
        print(f"  Best val accuracy : {bestValAcc*100:.2f}%  (epoch {bestEpoch})")
        print(f"  Total time        : {minutes}m {seconds}s")
        print(f"  Checkpoint saved  : {config.CNN_CHECKPOINT}")
        print("╚══════════════════════════════════════════════════════════════╝\n")

        historyPath = config.CHECKPOINTS_DIR / "trainingHistory.json"
        with open(historyPath, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"  Training history  : {historyPath}\n")

        checkpoint = torch.load(config.CNN_CHECKPOINT, map_location=self.device)
        self.model.load_state_dict(checkpoint["modelStateDict"])
        self.model.eval()

        return self.model


def runTraining() -> FlowerCNN:

    print("\n  Loading Oxford 102 dataset and building DataLoaders …")
    trainLoader, valLoader, _, idxToName = getDataLoaders()

    # Compute class weights from the full dataset for weighted loss
    from utils.dataLoader import FlowerDataset
    baseDataset  = FlowerDataset(config.ORGANIZED_DIR, transform=None)
    classWeights = baseDataset.getClassWeights()

    print(f"  Dataset loaded — {len(baseDataset)} total images, "
          f"{len(idxToName)} classes")
    print(f"  Class weight range: [{classWeights.min():.3f}, "
          f"{classWeights.max():.3f}]  (inverse frequency)")

    trainer = Trainer()
    model   = trainer.train(trainLoader, valLoader, classWeights)
    return model


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    print("\n╔══════════════════════════════════════════════╗")
    print("║   train/trainer.py — Sanity Check            ║")
    print("║   (2-epoch smoke test on synthetic data)     ║")
    print("╚══════════════════════════════════════════════╝")

    from torch.utils.data import TensorDataset

    device     = config.getDevice()
    numClasses = 10     # reduced for fast smoke test
    batchSize  = 8

    from torch.utils.data import Subset
    syntheticImages = torch.randn(80, 3, 224, 224)
    syntheticLabels = torch.randint(0, numClasses, (80,))
    syntheticDs     = TensorDataset(syntheticImages, syntheticLabels)

    trainLoader = DataLoader(Subset(syntheticDs, range(64)), batch_size=batchSize, shuffle=True)
    valLoader   = DataLoader(Subset(syntheticDs, range(64, 80)), batch_size=batchSize, shuffle=False)

    classWeights = torch.ones(numClasses)

    trainer = Trainer(
        device=device,
        numClasses=numClasses,
        numEpochs=2,
        patience=5,
    )

    import config as cfg
    originalCheckpoint  = cfg.CNN_CHECKPOINT
    cfg.CNN_CHECKPOINT  = cfg.CHECKPOINTS_DIR / "cnnFlower_smoketest.pth"

    model = trainer.train(trainLoader, valLoader, classWeights)

    cfg.CNN_CHECKPOINT = originalCheckpoint

    dummyInput = torch.randn(4, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummyInput)

    assert output.shape == (4, numClasses), f"Output shape mismatch: {output.shape}"
    assert len(trainer.history["trainLoss"]) == 2, "History not recorded correctly"
    print("  ✓ Output shape correct")
    print("  ✓ Training history recorded")
    print("  ✓ Smoke test passed\n")