"""Interactive food health classifier demo.

Run from the repository root:

    python interactive_demo/app.py

The server intentionally uses only the Python standard library for HTTP so the
demo can reuse the project's existing PyTorch dependencies without adding Flask,
Streamlit, or Gradio.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
from typing import Any
from urllib.parse import unquote, urlparse

from PIL import Image, UnidentifiedImageError


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR if (APP_DIR / "src" / "food_project").is_dir() else APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
if not STATIC_DIR.exists():
    STATIC_DIR = REPO_ROOT / "interactive_demo" / "static"
DEFAULT_CHECKPOINT_CANDIDATES = (
    APP_DIR / "best_model.pt",
    REPO_ROOT / "best_model.pt",
    APP_DIR / "model" / "best_model.pt",
    REPO_ROOT / "model" / "best_model.pt",
    REPO_ROOT
    / "result"
    / "classifier"
    / "resnet18_unfrozen_from_frozen_lr1e-4_128_e8"
    / "best_model.pt",
)
DEFAULT_CHECKPOINT = next(
    (path for path in DEFAULT_CHECKPOINT_CANDIDATES if path.exists()),
    DEFAULT_CHECKPOINT_CANDIDATES[-1],
)
DATA_URL_PATTERN = re.compile(r"^data:(?P<mime>[-\w.]+/[-\w.+]+)?;base64,(?P<data>.*)$")

sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    import torch

    from food_project.checkpointing import load_classifier_from_checkpoint
    from food_project.data import eval_transform
    from food_project.utils import get_device
except ImportError as exc:  # pragma: no cover - exercised when env is missing deps
    raise SystemExit(
        "Could not import the project ML dependencies. Activate the project "
        "environment first, then rerun this demo.\n"
        f"Original import error: {exc}"
    ) from exc


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    image_size: tuple[int, int]
    model_image_size: int
    device: str


class FoodHealthPredictor:
    def __init__(self, checkpoint_path: Path, device_name: str = "auto"):
        self.checkpoint_path = checkpoint_path
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}. Run the training "
                "pipeline or set CHECKPOINT_PATH to a trained classifier checkpoint."
            )

        self.device = get_device(device_name)
        self.model, self.checkpoint, self.class_names = load_classifier_from_checkpoint(
            self.checkpoint_path,
            device=self.device,
        )
        config = self.checkpoint.get("config", {})
        self.image_size = int(config.get("image_size", 128))
        self.transform = eval_transform(self.image_size)

    def predict(self, image_bytes: bytes) -> Prediction:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        original_size = image.size
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu()

        probabilities = {
            class_name: float(probs[index].item())
            for index, class_name in enumerate(self.class_names)
        }
        top_index = int(torch.argmax(probs).item())
        return Prediction(
            label=self.class_names[top_index],
            confidence=float(probs[top_index].item()),
            probabilities=probabilities,
            image_size=original_size,
            model_image_size=self.image_size,
            device=str(self.device),
        )


def parse_data_url(image_data: str) -> tuple[bytes, str]:
    match = DATA_URL_PATTERN.match(image_data)
    if match:
        mime_type = match.group("mime") or "application/octet-stream"
        payload = match.group("data")
    else:
        mime_type = "application/octet-stream"
        payload = image_data

    try:
        return base64.b64decode(payload, validate=True), mime_type
    except binascii.Error as exc:
        raise ValueError("Image payload is not valid base64.") from exc


def prediction_to_json(prediction: Prediction) -> dict[str, Any]:
    return {
        "label": prediction.label,
        "confidence": prediction.confidence,
        "probabilities": prediction.probabilities,
        "image_size": {
            "width": prediction.image_size[0],
            "height": prediction.image_size[1],
        },
        "model_image_size": prediction.model_image_size,
        "device": prediction.device,
    }


class DemoRequestHandler(BaseHTTPRequestHandler):
    predictor: FoodHealthPredictor
    max_upload_bytes: int

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib hook name
        parsed = urlparse(self.path)
        static_path = "index.html" if parsed.path in ("", "/") else parsed.path.lstrip("/")
        safe_path = Path(unquote(static_path))
        if safe_path.is_absolute() or ".." in safe_path.parts:
            self.send_response(HTTPStatus.BAD_REQUEST)
            self.end_headers()
            return

        file_path = (STATIC_DIR / safe_path).resolve()
        if not str(file_path).startswith(str(STATIC_DIR.resolve())) or not file_path.is_file():
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()
            return

        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_path.stat().st_size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - stdlib hook name
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self.send_json(
                {
                    "ok": True,
                    "checkpoint": str(self.predictor.checkpoint_path),
                    "classes": self.predictor.class_names,
                    "image_size": self.predictor.image_size,
                    "device": str(self.predictor.device),
                }
            )
            return

        static_path = "index.html" if parsed.path in ("", "/") else parsed.path.lstrip("/")
        self.serve_static(static_path)

    def do_POST(self) -> None:  # noqa: N802 - stdlib hook name
        parsed = urlparse(self.path)
        if parsed.path != "/api/predict":
            self.send_error_json(HTTPStatus.NOT_FOUND, "Unknown endpoint.")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            self.send_error_json(HTTPStatus.BAD_REQUEST, "Empty request body.")
            return
        if content_length > self.max_upload_bytes:
            self.send_error_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"Upload is too large. Limit is {self.max_upload_bytes // (1024 * 1024)} MB.",
            )
            return

        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            image_data = payload.get("image_data")
            if not isinstance(image_data, str) or not image_data:
                raise ValueError("Missing image_data.")
            image_bytes, mime_type = parse_data_url(image_data)
            if not mime_type.startswith("image/") and mime_type != "application/octet-stream":
                raise ValueError("Uploaded file is not an image.")

            prediction = self.predictor.predict(image_bytes)
            self.send_json(prediction_to_json(prediction))
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except UnidentifiedImageError:
            self.send_error_json(HTTPStatus.BAD_REQUEST, "Could not read the uploaded image.")
        except Exception as exc:  # pragma: no cover - safety net for demo server
            self.send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"Inference failed: {exc}",
            )

    def serve_static(self, relative_path: str) -> None:
        safe_path = Path(unquote(relative_path))
        if safe_path.is_absolute() or ".." in safe_path.parts:
            self.send_error_json(HTTPStatus.BAD_REQUEST, "Invalid path.")
            return

        file_path = (STATIC_DIR / safe_path).resolve()
        if not str(file_path).startswith(str(STATIC_DIR.resolve())) or not file_path.is_file():
            self.send_error_json(HTTPStatus.NOT_FOUND, "File not found.")
            return

        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        data = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_error_json(self, status: HTTPStatus, message: str) -> None:
        self.send_json({"ok": False, "error": message}, status=status)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"[demo] {self.address_string()} - {fmt % args}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("CHECKPOINT_PATH", DEFAULT_CHECKPOINT)),
        help="Path to a classifier checkpoint. Defaults to the best ResNet18 fine-tuned model.",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE", "auto"),
        help="auto, cpu, mps, cuda, or any torch device string.",
    )
    parser.add_argument(
        "--max-upload-mb",
        type=int,
        default=int(os.environ.get("MAX_UPLOAD_MB", "10")),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = FoodHealthPredictor(args.checkpoint.resolve(), device_name=args.device)

    handler_cls = type(
        "ConfiguredDemoRequestHandler",
        (DemoRequestHandler,),
        {
            "predictor": predictor,
            "max_upload_bytes": args.max_upload_mb * 1024 * 1024,
        },
    )
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    print("Food Health Lens demo")
    print(f"Serving: http://{args.host}:{args.port}")
    print(f"Checkpoint: {predictor.checkpoint_path}")
    print(f"Classes: {', '.join(predictor.class_names)}")
    print(f"Device: {predictor.device}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping demo server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
