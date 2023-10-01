# Copyright (c) 2023, Manfred Moitzi
# License: MIT License
import requests
import io
import base64
from PIL import Image

import a1111


def store_images(data: dict, prefix="output"):
    for num, image_str in enumerate(data["images"]):
        fname = a1111.SCRIPT_OUTPUT / f"{prefix}-{num}.png"
        image = Image.open(io.BytesIO(base64.b64decode(image_str)))
        image.save(fname)
        print(f"file '{fname}' saved")


def main(config: a1111.Config, batch_size: int):
    payload = a1111.Payload(
        prompt="woman",
        batch_size=batch_size,
        steps=20,
        width=512,
        height=512,
        seed=1,
    )
    checkpoint = config.find_checkpoint("photon")
    if checkpoint:
        print(checkpoint)
        payload.override(
            a1111.OverrideSettings(
                sd_model_checkpoint=checkpoint.model_name
            )
        )

    try:
        response = requests.post(url=a1111.API.TXT2IMG, json=payload.to_dict())
    except requests.ConnectionError:
        print("A1111 does not respond")
        return

    if response.status_code == 200:
        store_images(response.json())
    else:
        print(f"Error status code: {response.status_code}")


if __name__ == "__main__":
    if a1111.is_server_alive():
        _config = a1111.Config()
        _config.load()
        main(_config, 3)
    else:
        print("A1111 server not found")
