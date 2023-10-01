# Copyright (c) 2023, Manfred Moitzi
# License: MIT License
from __future__ import annotations
import pathlib
import requests
import dataclasses

# API help page: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API
SCRIPT_OUTPUT = pathlib.Path("../my-output")
PORT = 7860
LOCALHOST = "127.0.0.1"
URL = f"http://{LOCALHOST}:{PORT}"
API_V1 = URL + "/sdapi/v1/"


class API:
    TXT2IMG = API_V1 + "txt2img"
    IMG2IMG = API_V1 + "img2img"


class Samplers:
    DPMPP_2M_KARRAS = "DPM++ 2M Karras"
    DPMPP_SDE_KARRAS = "DPM++ SDE Karras"
    DPMPP_2M_SDE_KARRAS = "DPM++ 2M SDE Karras"
    DPMPP_2M_SDE_KARRAS_EXP = "DPM++ 2M SDE Exponential"
    EULER = "Euler"
    EULER_A = "Euler a"
    DPMPP_2S_A = "DPM++ 2S a"
    DPMPP_2M = "DPM++ 2M"
    DPMPP_SDE = "DPM++ SDE"
    DPMPP_2M_SDE = "DPM++ 2M SDE"
    DPMPP_2M_SDE_HEUN = "DPM++ 2M SDE Heun"
    DPMPP_2M_SDE_HEUN_KARRAS = "DPM++ 2M SDE Heun Karras"
    DPMPP_2M_SDE_HEUN_EXP = "DPM++ 2M SDE Heun Exponential"
    DPMPP_3M_SDE = "DPM++ 3M SDE"
    DPMPP_3M_SDE_KARRAS = "DPM++ 3M SDE Karras"
    DPMPP_3M_SDE_EXP = "DPM++ 3M SDE Exponential"
    DPMPP_2S_A_KARRAS = "DPM++ 2S a Karras"
    UniPC = "UniPC"


# show docs: URL/docs
# show txt2img docs: URL/sdapi/v1/txt2img
# show imp2img docs: URL/sdapi/v1/img2img


@dataclasses.dataclass
class OverrideSettings:
    sd_model_checkpoint: str = ""  # checkpoint_title

    def to_dict(self) -> dict[str, str | int]:
        return {"sd_model_checkpoint": self.sd_model_checkpoint}


@dataclasses.dataclass
class Payload:
    prompt: str = ""
    negative_prompt: str = ""
    sampler_name: str = Samplers.DPMPP_2M_KARRAS
    batch_count: int = 1
    batch_size: int = 1
    steps: int = 20
    cfg_scale: int = 7
    width: int = 512
    height: int = 512
    seed: int = 1
    _override: OverrideSettings | None = None

    def override(self, data: OverrideSettings):
        self._override = data

    def to_dict(self) -> dict[str, str | int]:
        data = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "sampler_name": self.sampler_name,
            "batch_size": self.batch_size,
            "n_iter": self.batch_count,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
        }
        if self._override:
            data["override_settings"] = self._override.to_dict()
        return data


def is_server_alive() -> bool:
    try:
        respond = requests.get(url=f"{URL}/app_id")
    except requests.ConnectionError:
        return False
    return respond.status_code == 200


@dataclasses.dataclass
class Checkpoint:
    title: str = ""
    model_name: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> Checkpoint:
        return cls(
            title=data.get("title", ""),
            model_name=data.get("model_name", ""),
        )

    @property
    def is_sd15(self) -> bool:
        return self.model_name.startswith("SD15")

    @property
    def is_sdxl(self) -> bool:
        return self.model_name.startswith("SDXL")


class Config:
    def __init__(self) -> None:
        self.checkpoints: list[Checkpoint] = []

    def load(self) -> None:
        self.query_checkpoints()

    def query_checkpoints(self) -> None:
        try:
            response = requests.get(url=f"{API_V1}sd-models")
        except requests.ConnectionError:
            print("HTTP connection error")
        else:
            self.checkpoints.clear()
            self.checkpoints.extend(Checkpoint.from_dict(d) for d in response.json())

    def find_checkpoint(self, name: str) -> Checkpoint | None:
        name = name.lower()
        for model in self.checkpoints:
            if name in model.model_name.lower():
                return model
        return None
