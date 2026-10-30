"""Diagnostic only: export embedded textures near the Scene201 kettle."""
import bpy
from pathlib import Path

target = Path(__file__).resolve().parent / "diagnostic_textures"
target.mkdir(exist_ok=True)
for name in ("Image_401", "Image_402"):
    image = bpy.data.images.get(name)
    if image is not None:
        image.save_render(str(target / f"{name}.png"))
        print(f"EXPORTED={target / (name + '.png')}")
