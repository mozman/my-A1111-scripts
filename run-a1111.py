# Copyright (c) 2023, Manfred Moitzi
# License: MIT License
import requests
import io
import base64
from PIL import Image

import a1111


def main(config: a1111.Config):
    payload = {
        "prompt": "woman",
        "sampler_name": "DPM++ 2M Karras",
        "batch_size": 1,
        "steps": 20,
        "cfg_scale": 7,
        "width": 512,
        "height": 512,
        "restore_faces": False,
        "seed": 1,
    }
    override_settings = {
        "sd_model_checkpoint": config.get_checkpoint_title("photon"),
    }
    payload["override_settings"] = override_settings

    try:
        response = requests.post(url=f"{a1111.URL}/sdapi/v1/txt2img", json=payload)
    except requests.ConnectionError:
        return
    r = response.json()

    image = Image.open(io.BytesIO(base64.b64decode(r["images"][0])))
    image.save(a1111.SCRIPT_OUTPUT / "output.png")


if __name__ == "__main__":
    if a1111.is_server_alive():
        _config = a1111.Config()
        _config.setup()
        main(_config)
    else:
        print("A1111 server not found")
