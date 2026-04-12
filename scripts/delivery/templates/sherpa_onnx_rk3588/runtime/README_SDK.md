# RKVoice sherpa-onnx Runtime Bundle

This runtime bundle packages a board-side offline ASR + TTS baseline for RK3588:

- bin/sherpa-onnx-offline: offline ASR executable.
- bin/sherpa-onnx-offline-tts: offline TTS executable.
- lib/libsherpa-onnx-c-api.so: shared library from the prebuilt sherpa-onnx runtime.
- models/asr/cpu/sense-voice/: CPU ONNX SenseVoice model.
- models/asr/rknn/sense-voice-rk3588-20s/: RKNN SenseVoice model for RK3588 validation.
- models/tts/vits-icefall-zh-aishell3/: Chinese VITS TTS baseline.
- run_asr.sh, run_tts.sh, smoketest.sh: board-side helper entrypoints.
- tools/check_rknn_env.sh: verifies librknnrt.so visibility and basic RKNN runtime state.
- tools/profile_asr_inference.sh: captures NPU load samples while running the RKNN ASR path.
- tools/profile_tts_inference.sh: captures RSS, CPU ticks, and RK NPU load samples while running the TTS path.

Backend status in this bundle:

- ASR CPU/ONNX: available.
- ASR RKNN/NPU: available when the board provides a compatible librknnrt.so.
- TTS CPU/ONNX: available.
- TTS RKNN/NPU: not the default target in this delivery path.

Typical usage on the board:

    ./run_asr.sh
    ./run_tts.sh "短波电台链路测试完成。"
    ./smoketest.sh "你好，欢迎使用 RKVoice sherpa-onnx 离线语音服务。" ./output/smoke_test_tts.wav

Default TTS assets are included as a technical baseline. Verify upstream model and dataset licensing before commercial shipment.