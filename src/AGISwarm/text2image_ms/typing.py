"""
This module contains the typing classes for the Text2Image Pipeline.

"""

from dataclasses import dataclass

from pydantic import BaseModel
from uvicorn.config import LoopSetupType


@dataclass
class DiffusionConfig(BaseModel):
    """
    A class to hold the configuration for the Diffusion Pipeline initialization.
    """

    dtype: str
    device: str
    low_cpu_mem_usage: bool


@dataclass
class UvicornConfig(BaseModel):
    """
    A class to hold the configuration for the Uvicorn.
    """

    host: str
    port: int
    log_level: str
    loop: LoopSetupType


@dataclass
class GUIConfig(BaseModel):
    """
    A class to hold the configuration for the GUI.
    """

    latent_update_frequency: int


@dataclass
class Config(BaseModel):
    """
    A class to hold the configuration for the Text2Image Pipeline.
    """

    hf_model_name: str
    t2i_model_config: DiffusionConfig
    gui_config: GUIConfig
    uvicorn_config: UvicornConfig


class Text2ImageGenerationConfig(BaseModel):
    """
    A class to hold the configuration for the Text2Image Pipeline generation.
    """

    prompt: str
    negative_prompt: str
    num_inference_steps: int
    guidance_scale: float
    seed: int
    width: int
    height: int
