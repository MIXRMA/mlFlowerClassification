import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class ConvBlock(nn.Module):
  
    def __init__(self, inChannels: int, outChannels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inChannels, outChannels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True),

            nn.Conv2d(outChannels, outChannels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FlowerCNN(nn.Module):

    def __init__(
        self,
        numClasses:   int       = config.NUM_CLASSES,
        convChannels: list      = None,
        fcHiddenDim:  int       = config.FC_HIDDEN_DIM,
        dropoutRate:  float     = config.DROPOUT_RATE,
    ):
        super().__init__()

        if convChannels is None:
            convChannels = config.CONV_CHANNELS

        inCh = 3
        convBlocks = []
        for outCh in convChannels:
            convBlocks.append(ConvBlock(inCh, outCh))
            inCh = outCh

        self.features = nn.Sequential(*convBlocks)

        self.pool = nn.AdaptiveAvgPool2d(output_size=(4, 4))

        flattenedDim = convChannels[-1] * 4 * 4

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(flattenedDim, fcHiddenDim, bias=False),
            nn.BatchNorm1d(fcHiddenDim),
            nn.ReLU(inplace=True),

            nn.Dropout(p=dropoutRate),
            nn.Linear(fcHiddenDim, numClasses),
        )

        self._initWeights()

    def _initWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias,   0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

    def getFeatureEmbedding(self, x: torch.Tensor) -> torch.Tensor:
     
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        fc1    = self.classifier[1]
        bn1    = self.classifier[2]
        relu1  = self.classifier[3]
        return relu1(bn1(fc1(x)))

    def countParameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    device = config.getDevice()
    model  = FlowerCNN().to(device)

    dummyBatch = torch.randn(4, 3, 224, 224).to(device)

    model.eval()
    with torch.no_grad():
        logits    = model(dummyBatch)
        embedding = model.getFeatureEmbedding(dummyBatch)
        probs     = F.softmax(logits, dim=1)

    print("\n╔══════════════════════════════════════════════╗")
    print("║   models/cnn.py — Sanity Check               ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  Device              : {device}")
    print(f"  Input shape         : {tuple(dummyBatch.shape)}")
    print(f"  Logits shape        : {tuple(logits.shape)}   ← (B, 102)")
    print(f"  Embedding shape     : {tuple(embedding.shape)}  ← (B, 512)")
    print(f"  Softmax probs sum   : {probs.sum(dim=1).tolist()}  ← all ≈ 1.0")
    print(f"  Trainable params    : {model.countParameters():,}")
    print()

    print("  ── Architecture ──────────────────────────────")
    for name, module in model.named_children():
        print(f"  {name:<14}: {module.__class__.__name__}")
    print()

    assert logits.shape    == (4, 102),  "Logit shape mismatch"
    assert embedding.shape == (4, 512),  "Embedding shape mismatch"
    assert torch.allclose(probs.sum(dim=1), torch.ones(4).to(device), atol=1e-5), \
        "Softmax probabilities do not sum to 1"
    print("  ✓ All assertions passed\n")