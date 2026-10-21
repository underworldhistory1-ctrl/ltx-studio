"""Shared-device queue admission for the LTX ComfyUI worker (explicit opt-in)."""
from server import PromptServer
from .shared_gpu_guard import install_from_environment

install_from_environment(PromptServer.instance.prompt_queue)
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
