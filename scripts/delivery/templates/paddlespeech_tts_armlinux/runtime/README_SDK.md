# RKVoice Binary SDK

This runtime bundle ships a basic binary SDK layout for board-side offline TTS:

- bin/rkvoice_tts_demo: reference executable demo.
- bin/paddlespeech_tts_demo: compatibility executable kept for existing scripts.
- lib/librkvoice_tts.so: shared library for dynamic linking.
- lib/librkvoice_tts.a: static wrapper archive for static linking.
- include/rkvoice_tts_api.h: stable C API header.
- examples/c_api_demo.c: minimal sample showing how to call the C API.

Current backend support in this bundle:

- cpu: available.
- rknn: reserved for the RK3588 NPU delivery path and not compiled into this bundle yet.

Typical demo invocation:

    ./bin/rkvoice_tts_demo --sentence "短波电台指令测试。" --output_wav ./output/demo.wav

Typical dynamic linking compile line on the board:

    gcc ./examples/c_api_demo.c -I./include -L./lib -lrkvoice_tts -Wl,-rpath,'$ORIGIN/../lib' -o ./output/c_api_demo
