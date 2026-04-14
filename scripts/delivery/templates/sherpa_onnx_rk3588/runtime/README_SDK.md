# RKVoice sherpa-onnx ASR Component

This component packages the board-side streaming ASR (RKNN) subtree inside the unified RKVoice runtime project for RK3588:

- bin/sherpa-onnx: streaming ASR executable (online/transducer).
- lib/libsherpa-onnx-c-api.so: shared library from the prebuilt sherpa-onnx runtime.
- audios/: real speech WAV files used as default ASR test inputs.
- models/asr/streaming-rknn/streaming-zipformer-rk3588-small/: streaming Zipformer RKNN transducer model (bilingual zh-en).
- run_asr.sh, smoketest.sh: board-side helper entrypoints. `smoketest.sh` defaults to the first WAV under audios/ and logs elapsed seconds plus RTF for report parsing.
- tools/check_rknn_env.sh: verifies librknnrt.so visibility and basic RKNN runtime state.
- tools/profile_asr_inference.sh: captures RKNN runtime layer logs via RKNN_LOG_LEVEL=4 plus rknpu/load samples while running ASR.

If you place rknn_eval_perf.txt, rknn_query_perf_detail.txt, rknn_perf_run.json, or rknn_memory_profile.txt under output/, the integrated HTML report will prioritize those official RKNN profiler artifacts over the fallback heatmaps.

Backend status in this component:

- ASR streaming RKNN (default): available via streaming Zipformer RKNN transducer model.

TTS is provided by the sibling tts/ component under the same unified runtime root.

Typical usage on the board:

    # Streaming ASR (RKNN) — default
    ./run_asr.sh

    # Full smoke test (defaults to the first WAV under audios/)
    ./smoketest.sh

Environment variables for run_asr.sh:

    RKVOICE_ASR_NUM_THREADS=2              (default: 2)
    RKVOICE_ASR_RKNN_NUM_THREADS=2         (default: same as NUM_THREADS, controls NPU core)