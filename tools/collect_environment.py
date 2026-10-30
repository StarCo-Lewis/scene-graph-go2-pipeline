"""Collect selected dependency versions, never credentials or all environment variables."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packages = {}
    for name in ("numpy", "Pillow", "torch", "torchvision", "sentence-transformers",
                 "transformers", "open_clip_torch", "open3d", "scipy", "PyYAML",
                 "opencv-python", "huggingface-hub", "hydra-core", "rclpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {"python": sys.version, "executable": sys.executable,
              "system": platform.system(), "release": platform.release(),
              "packages": packages,
              "note": "Null means distribution metadata unavailable, not proof of absence. No GPU task, network call or environment-variable dump performed."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=True, indent=2)
    print(args.output)


if __name__ == "__main__":
    main()
