# Copyright (c) 2023, Manfred Moitzi
# License: MIT License
from __future__ import annotations
from typing import Iterator, Iterable
import requests
import io
import base64
from PIL import Image

import a1111

CKPT_TEST = "ckpt-test"

# Check Issues:
# https://civitai.com/articles/260/can-we-identify-most-stable-diffusion-model-issues-with-just-a-few-circles
COMMON_NEG = (
    "ugly, old, mutation, low quality, doll, long neck, text, signature, artist "
    "name, bad anatomy, poorly drawn, malformed, deformed, blurry, out of focus, noise, dust"
)
TESTS = {
    "jenifer_lawrence": [
        "photo of (Jennifer Lawrence:0.9) beautiful young professional photo high quality highres makeup",
        COMMON_NEG,
    ],
    "woman_photo": [
        "photo of woman standing full body beautiful young professional photo high quality highres makeup",
        COMMON_NEG,
    ],
    "naked_woman": [
        "photo of naked woman sexy beautiful young professional photo high quality highres makeup",
        COMMON_NEG,
    ],
    "streets": [
        "photo of city detailed streets roads buildings professional photo high quality highres",
        COMMON_NEG,
    ],
    "circle": [
        "minimalism simple illustration vector art style clean single black circle inside "
        "white rectangle symmetric shape sharp professional print quality highres high contrast black and white",
        COMMON_NEG,
    ],
    "man_photo": [
        "photo of man standing full body beautiful young professional photo high quality highres",
        COMMON_NEG,
    ],
    "woman": ["woman", ""],
    "man": ["man", ""],
}


def image_grid(
    images: Iterable[Image],
    tile_size: tuple[int, int],
    grid_size: tuple[int, int],
) -> Image:
    row_count, col_count = grid_size
    tile_width, tile_height = tile_size

    grid = Image.new("RGB", size=(col_count * tile_width, row_count * tile_height))
    last_grid_index = row_count * col_count - 1
    for index, image in enumerate(images):
        if index > last_grid_index:
            break
        grid.paste(
            image,
            box=(index % col_count * tile_width, index // col_count * tile_height),
        )
    return grid


def decode_images(images: list[str]) -> Iterator[Image]:
    return (Image.open(io.BytesIO(base64.b64decode(image))) for image in images)


def run_test(
    name: str, checkpoint: a1111.Checkpoint, prompt: str, negative_prompt: str = ""
):
    folder = a1111.SCRIPT_OUTPUT / CKPT_TEST / name
    if not folder.exists():
        folder.mkdir(parents=True)
    png_name = folder / f"{checkpoint.model_name}-{name}.png"
    if png_name.exists():
        print(
            f"skipping test for checkpoint: {checkpoint.model_name}, testfile already exist"
        )
        return

    print(f"testing checkpoint: {checkpoint.model_name}")
    payload = a1111.Payload(
        prompt=prompt,
        negative_prompt=negative_prompt,
        batch_count=9,
        steps=20,
        width=512,
        height=512,
        seed=1,
    )
    if checkpoint.is_sdxl:
        payload.width = 1024
        payload.height = 1024
    payload.override(a1111.OverrideSettings(sd_model_checkpoint=checkpoint.model_name))

    try:
        response = requests.post(url=a1111.API.TXT2IMG, json=payload.to_dict())
    except requests.ConnectionError:
        print("A1111 does not respond")
        return

    if response.status_code == 200:
        data = response.json()
        images: list[str] = data["images"]
        grid = image_grid(
            decode_images(images),
            tile_size=(payload.width, payload.height),
            grid_size=(3, 3),
        )
        grid.save(png_name)
        print(f"image grid '{png_name}' saved")
    else:
        print(f"Error status code: {response.status_code}")


def main(config: a1111.Config, *, tests: list[str]):
    for name in tests:
        try:
            prompt, negative_prompt = TESTS[name]
        except KeyError:
            print(f"invalid test name: {name}")
            continue
        print(f"\nLaunching checkpoint test: {name}")
        print(f"prompt: {prompt}")
        print("-" * 79)
        for checkpoint in config.checkpoints:
            if "refiner" in checkpoint.model_name.lower():
                continue
            try:
                run_test(
                    name, checkpoint, prompt=prompt, negative_prompt=negative_prompt
                )
            except requests.ConnectionError:
                print("A1111 does not respond")


if __name__ == "__main__":
    if a1111.is_server_alive():
        _config = a1111.Config()
        _config.load()
        main(_config, tests=["circle"])
    else:
        print("A1111 server not running")
