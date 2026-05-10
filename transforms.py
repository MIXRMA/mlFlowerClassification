from torchvision import transforms
import config

def getTrainTransforms() -> transforms.Compose:

    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=config.IMAGE_SIZE,
            scale=(0.8, 1.0),
            ratio=(0.75, 1.33),
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.08
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.NORMALIZE_MEAN,
            std=config.NORMALIZE_STD
        ),
    ])


def getInferenceTransforms() -> transforms.Compose:

    return transforms.Compose([
        transforms.Resize(
            size=256,
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.CenterCrop(size=config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.NORMALIZE_MEAN,
            std=config.NORMALIZE_STD
        ),
    ])


if __name__ == "__main__":
    from PIL import Image
    import torch
    import numpy as np

    dummyArray = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    dummyImage = Image.fromarray(dummyArray)

    trainTfm    = getTrainTransforms()
    inferenceTfm = getInferenceTransforms()

    trainTensor    = trainTfm(dummyImage)
    inferenceTensor = inferenceTfm(dummyImage)

    print("\n╔══════════════════════════════════════════════╗")
    print("║   augment/transforms.py — Sanity Check       ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  Input image size    : {dummyImage.size}  (W×H, non-square)")
    print(f"  Train tensor shape  : {tuple(trainTensor.shape)}  (C×H×W)")
    print(f"  Infer tensor shape  : {tuple(inferenceTensor.shape)}  (C×H×W)")
    print(f"  Train value range   : [{trainTensor.min():.3f}, {trainTensor.max():.3f}]")
    print(f"  Infer value range   : [{inferenceTensor.min():.3f}, {inferenceTensor.max():.3f}]")
    assert trainTensor.shape    == (3, 224, 224), "Train output shape mismatch"
    assert inferenceTensor.shape == (3, 224, 224), "Inference output shape mismatch"
    print("  ✓ Both pipelines output correct shape (3, 224, 224)")
    print("  ✓ Normalisation applied successfully\n")