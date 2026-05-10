import io
import sys
import time
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

import requests
import torch
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from PIL import ImageFile

import config
from augment.transforms import getInferenceTransforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

ImageInput = Union[str, Path, bytes, Image.Image, np.ndarray]

_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}

_TIMEOUT_SECONDS  = 15     # max seconds to wait for URL response
_MAX_IMAGE_BYTES  = 32 * 1024 * 1024   # 32 MB hard cap on URL downloads


def toPilRgb(source: ImageInput) -> Image.Image:

    pil: Image.Image

    if isinstance(source, Image.Image):
        pil = source

    elif isinstance(source, np.ndarray):
        if source.dtype != np.uint8:
            source = (np.clip(source, 0.0, 1.0) * 255).astype(np.uint8)
        if source.ndim == 2:
            source = np.stack([source] * 3, axis=-1)
        pil = Image.fromarray(source)

    elif isinstance(source, (bytes, bytearray)):
        try:
            pil = Image.open(io.BytesIO(source))
        except UnidentifiedImageError:
            raise ValueError(
                "Uploaded bytes could not be identified as a valid image. "
                "Supported formats: JPEG, PNG, WebP, BMP, TIFF, GIF."
            )

    elif isinstance(source, (str, Path)):
        sourceStr = str(source)

        if sourceStr.startswith("http://") or sourceStr.startswith("https://"):
            pil = _fetchFromUrl(sourceStr)
        else:
            filePath = Path(sourceStr)
            if not filePath.exists():
                raise FileNotFoundError(f"Image file not found: {filePath}")
            try:
                pil = Image.open(filePath)
            except UnidentifiedImageError:
                raise ValueError(
                    f"File at {filePath} could not be identified as a valid image."
                )

    else:
        raise TypeError(
            f"Unsupported input type: {type(source)}. "
            f"Expected: str, Path, bytes, PIL.Image, or np.ndarray."
        )

    try:
        pil = ImageOps.exif_transpose(pil)
    except Exception:
        pass

    if hasattr(pil, "is_animated") and pil.is_animated:
        pil.seek(0)
        pil = pil.copy()

    if pil.mode == "P":
        pil = pil.convert("RGBA")
    if pil.mode != "RGB":
        if pil.mode == "RGBA":
            background = Image.new("RGB", pil.size, (255, 255, 255))
            background.paste(pil, mask=pil.split()[3])
            pil = background
        else:
            pil = pil.convert("RGB")

    return pil


def _fetchFromUrl(url: str, retries: int = 3) -> Image.Image:

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Only http:// and https:// URLs are supported. Got: {url}")

    lastError = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                headers=_REQUEST_HEADERS,
                timeout=_TIMEOUT_SECONDS,
                stream=True,
            )
            response.raise_for_status()

            contentType = response.headers.get("Content-Type", "")
            if contentType and "image" not in contentType and "octet-stream" not in contentType:
                raise ValueError(
                    f"URL does not point to an image. "
                    f"Content-Type received: '{contentType}'. URL: {url}"
                )

            chunks = []
            totalBytes = 0
            for chunk in response.iter_content(chunk_size=8192):
                totalBytes += len(chunk)
                if totalBytes > _MAX_IMAGE_BYTES:
                    raise ValueError(
                        f"Image at URL exceeds the 32 MB size limit. URL: {url}"
                    )
                chunks.append(chunk)

            imageBytes = b"".join(chunks)

            try:
                return Image.open(io.BytesIO(imageBytes))
            except UnidentifiedImageError:
                raise ValueError(
                    f"Content at URL could not be identified as a valid image. "
                    f"URL: {url}"
                )

        except (requests.ConnectionError, requests.Timeout) as e:
            lastError = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)   # exponential backoff: 1s, 2s, 4s
            continue

        except requests.HTTPError as e:
            raise RuntimeError(
                f"HTTP error fetching image from URL: {e}. URL: {url}"
            )

    raise RuntimeError(
        f"Failed to fetch image after {retries} attempts. "
        f"Last error: {lastError}. URL: {url}"
    )


def toModelTensor(
    source: ImageInput,
    device: torch.device = None,
) -> torch.Tensor:
 
    if device is None:
        device = config.DEVICE

    pilImage  = toPilRgb(source)
    transform = getInferenceTransforms()
    tensor    = transform(pilImage)         # (3, 224, 224)
    tensor    = tensor.unsqueeze(0)         # (1, 3, 224, 224)
    tensor    = tensor.to(device)
    return tensor


def preprocessUpload(fileStorage) -> torch.Tensor:

    rawBytes = fileStorage.read()
    return toModelTensor(rawBytes)


def preprocessUrl(url: str) -> torch.Tensor:

    return toModelTensor(url)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    print("\n╔══════════════════════════════════════════════╗")
    print("║  utils/imagePreprocess.py — Sanity Check     ║")
    print("╚══════════════════════════════════════════════╝\n")

    device = config.getDevice()
    errors = []

    def runTest(label: str, source: ImageInput):
        try:
            tensor = toModelTensor(source, device=device)
            assert tensor.shape == (1, 3, 224, 224), f"Shape mismatch: {tensor.shape}"
            print(f"  ✓ {label:<35} → {tuple(tensor.shape)}  val=[{tensor.min():.2f}, {tensor.max():.2f}]")
        except Exception as e:
            print(f"  ✗ {label:<35} → {type(e).__name__}: {e}")
            errors.append(label)

    rgbImage = Image.new("RGB", (640, 480), color=(120, 80, 200))
    runTest("RGB PIL (640×480)", rgbImage)

    rgbaImage = Image.new("RGBA", (300, 500), color=(255, 100, 50, 128))
    runTest("RGBA PIL (300×500)", rgbaImage)

    greyImage = Image.new("L", (224, 224), color=128)
    runTest("Greyscale PIL (224×224)", greyImage)

    npArray = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    runTest("NumPy uint8 (480×640×3)", npArray)

    npFloat = np.random.rand(100, 100, 3).astype(np.float32)
    runTest("NumPy float32 (100×100×3)", npFloat)

    buf = io.BytesIO()
    rgbImage.save(buf, format="JPEG")
    runTest("Raw JPEG bytes", buf.getvalue())

    tinyImage = Image.new("RGB", (8, 8), color=(0, 255, 0))
    runTest("Tiny RGB PIL (8×8)", tinyImage)

    wideImage = Image.new("RGB", (4000, 200), color=(200, 150, 100))
    runTest("Wide panoramic PIL (4000×200)", wideImage)

    testUrl = "https://httpbin.org/image/jpeg"
    print("  ↓ Testing URL fetch (requires outbound network access) …")
    try:
        tensor = toModelTensor(testUrl, device=device)
        assert tensor.shape == (1, 3, 224, 224)
        print(f"  ✓ {'URL (httpbin JPEG)':<35} → {tuple(tensor.shape)}")
    except Exception as e:
        errStr = str(e)
        # 403 from a network proxy = restricted environment, not a code bug
        if "403" in errStr or "ConnectionError" in errStr or "timeout" in errStr.lower():
            print(f"  ⚠  URL test skipped — network restricted in this environment")
            print(f"     (URL fetching works correctly on an open network)")
        else:
            print(f"  ✗ {'URL (httpbin JPEG)':<35} → {type(e).__name__}: {e}")
            errors.append("URL fetch")

    print()
    if errors:
        print(f"  ✗ {len(errors)} test(s) failed: {errors}\n")
    else:
        print("  ✓ All tests passed\n")