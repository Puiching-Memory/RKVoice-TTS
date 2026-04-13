# RKVoice sherpa-onnx Runtime Bundle

This runtime bundle packages a board-side streaming ASR + offline ASR + TTS baseline for RK3588:

- bin/sherpa-onnx: streaming ASR executable (online/transducer).
- bin/sherpa-onnx-offline: offline ASR executable.
- bin/sherpa-onnx-offline-tts: offline TTS executable.
- lib/libsherpa-onnx-c-api.so: shared library from the prebuilt sherpa-onnx runtime.
- models/asr/streaming/streaming-zipformer-multi-zh-hans/: streaming Zipformer transducer model (INT8, Chinese).
- models/asr/cpu/sense-voice/: CPU ONNX SenseVoice model (offline fallback).
- models/asr/rknn/sense-voice-rk3588-20s/: RKNN SenseVoice model for RK3588 validation.
- models/tts/vits-icefall-zh-aishell3/: Chinese VITS TTS baseline.
- run_asr.sh, run_tts.sh, smoketest.sh: board-side helper entrypoints.
- tools/check_rknn_env.sh: verifies librknnrt.so visibility and basic RKNN runtime state.
- tools/profile_asr_inference.sh: captures RKNN runtime layer logs via RKNN_LOG_LEVEL=4 plus rknpu/load samples while running the RKNN ASR path.
- tools/profile_tts_inference.sh: captures RSS, CPU ticks, and RK NPU load samples while running the TTS path.

If you place rknn_eval_perf.txt, rknn_query_perf_detail.txt, rknn_perf_run.json, or rknn_memory_profile.txt under output/, the integrated HTML report will prioritize those official RKNN profiler artifacts over the fallback heatmaps.

Backend status in this bundle:

- ASR streaming (default): available via streaming Zipformer transducer model.
- ASR CPU/ONNX offline: available via SenseVoice.
- ASR RKNN/NPU offline: available when the board provides a compatible librknnrt.so.
- TTS CPU/ONNX: available.
- TTS RKNN/NPU: not the default target in this delivery path.

Typical usage on the board:

    # Streaming ASR (default)
    ./run_asr.sh

    # Offline ASR (CPU)
    RKVOICE_ASR_MODE=offline RKVOICE_ASR_PROVIDER=cpu ./run_asr.sh

    # Offline ASR (RKNN)
    RKVOICE_ASR_MODE=offline RKVOICE_ASR_PROVIDER=rknn ./run_asr.sh

    # TTS
    ./run_tts.sh "短波电台链路测试完成。"

    # Full smoke test
    ./smoketest.sh "你好，欢迎使用 RKVoice sherpa-onnx 离线语音服务。" ./output/smoke_test_tts.wav

Environment variables for run_asr.sh:

    RKVOICE_ASR_MODE=streaming|offline   (default: streaming)
    RKVOICE_ASR_PROVIDER=cpu|rknn        (offline mode only, default: cpu)
    RKVOICE_ASR_NUM_THREADS=2            (default: 2)

Default TTS assets are included as a technical baseline. Verify upstream model and dataset licensing before commercial shipment.