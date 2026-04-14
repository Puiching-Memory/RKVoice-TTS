from __future__ import annotations

from importlib import import_module

__all__ = [
    "ASR_DEFAULT_CACHE_DIR",
    "ASR_DEFAULT_REMOTE_DIR",
    "ASR_DEFAULT_RUNTIME_DIR",
    "ASR_DEFAULT_STAGE_DIR",
    "AUDIOS_DIR",
    "CONFIG_LOCAL_DIR",
    "TTS_DEFAULT_CACHE_DIR",
    "TTS_DEFAULT_REMOTE_DIR",
    "TTS_DEFAULT_RUNTIME_DIR",
    "TTS_DEFAULT_STAGE_DIR",
    "TTS_DEFAULT_TEXT",
    "asr",
    "guess_source_ip",
    "load_local_settings",
    "main",
    "open_ssh_client",
    "parse_args",
    "resolve_int_option",
    "resolve_path_option",
    "resolve_required_text_option",
    "resolve_text_option",
    "sh_quote",
    "tts",
]

_EXPORTS = {
    "ASR_DEFAULT_CACHE_DIR": ".config",
    "ASR_DEFAULT_REMOTE_DIR": ".config",
    "ASR_DEFAULT_RUNTIME_DIR": ".config",
    "ASR_DEFAULT_STAGE_DIR": ".config",
    "AUDIOS_DIR": ".config",
    "CONFIG_LOCAL_DIR": ".config",
    "TTS_DEFAULT_CACHE_DIR": ".config",
    "TTS_DEFAULT_REMOTE_DIR": ".config",
    "TTS_DEFAULT_RUNTIME_DIR": ".config",
    "TTS_DEFAULT_STAGE_DIR": ".config",
    "TTS_DEFAULT_TEXT": ".config",
    "asr": ".asr",
    "guess_source_ip": ".remote",
    "load_local_settings": ".config",
    "main": ".cli",
    "open_ssh_client": ".remote",
    "parse_args": ".cli",
    "resolve_int_option": ".config",
    "resolve_path_option": ".config",
    "resolve_required_text_option": ".config",
    "resolve_text_option": ".config",
    "sh_quote": ".remote",
    "tts": ".tts",
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
