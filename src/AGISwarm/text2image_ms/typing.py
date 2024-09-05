"""
This module contains the typing classes for the Text2Image Pipeline.

"""

from dataclasses import dataclass

from pydantic import BaseModel


@dataclass
class DiffusionConfig(BaseModel):
    """
    A class to hold the configuration for the Diffusion Pipeline initialization.
    """

    model: str
    dtype: str
    device: str
    safety_checker: str | None
    requires_safety_checker: bool
    low_cpu_mem_usage: bool


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

    diffusion_config: DiffusionConfig
    gui_config: GUIConfig



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
