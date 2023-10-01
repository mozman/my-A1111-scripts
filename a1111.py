# Copyright (c) 2023, Manfred Moitzi
# License: MIT License
from __future__ import annotations
import pathlib
import requests

SCRIPT_OUTPUT = pathlib.Path("../my-output")
URL = "http://127.0.0.1:7860"
URL_API = URL + "/sdapi/v1/"

# show docs: URL/docs
TXT2IMG_PAYLOAD_DEFAULT = {
    "prompt": "",
    "negative_prompt": "",
    "styles": tuple(),
    "seed": -1,
    "subseed": -1,
    "subseed_strength": 0,
    "seed_resize_from_h": -1,
    "seed_resize_from_w": -1,
    "sampler_name": "string",
    "batch_size": 1,
    "n_iter": 1,
    "steps": 20,
    "cfg_scale": 7,
    "width": 512,
    "height": 512,
    "restore_faces": False,
    "tiling": False,
    "do_not_save_samples": False,
    "do_not_save_grid": False,
    "eta": 0,
    "denoising_strength": 0,
    "s_min_uncond": 0,
    "s_churn": 0,
    "s_tmax": 0,
    "s_tmin": 0,
    "s_noise": 0,
    "override_settings": {},
    "override_settings_restore_afterwards": True,
    "refiner_checkpoint": "",
    "refiner_switch_at": 0,
    "disable_extra_networks": False,
    "comments": {},
    "enable_hr": False,
    "firstphase_width": 0,
    "firstphase_height": 0,
    "hr_scale": 2,
    "hr_upscaler": "string",
    "hr_second_pass_steps": 0,
    "hr_resize_x": 0,
    "hr_resize_y": 0,
    "hr_checkpoint_name": "string",
    "hr_sampler_name": "string",
    "hr_prompt": "",
    "hr_negative_prompt": "",
    "sampler_index": "Euler",
    "script_name": "string",
    "script_args": [],
    "send_images": True,
    "save_images": False,
    "alwayson_scripts": {},
}


OVERRIDE_SETTINGS_EXAMPLE = {
    "sd_model_checkpoint": "checkpoint_title",
}


def is_server_alive() -> bool:
    try:
        requests.get(url=f"{URL}/docs")
    except requests.ConnectionError:
        return False
    return True


class Config:
    def __init__(self) -> None:
        self.checkpoints: list[dict] = []

    def setup(self) -> None:
        self.query_checkpoints()

    def query_checkpoints(self) -> None:
        try:
            response = requests.get(url=f"{URL_API}sd-models")
        except requests.ConnectionError:
            print("HTTP connection error")
        else:
            self.checkpoints.clear()
            self.checkpoints.extend(response.json())

    def get_checkpoint_title(self, name: str) -> str:
        for model in self.checkpoints:
            if name in model.get("model_name", ""):
                return model.get("title", "")
        return ""
