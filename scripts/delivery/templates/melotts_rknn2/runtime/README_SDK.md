# RKVoice MeloTTS-RKNN2 Runtime

This runtime bundle ships a Python-based MeloTTS-RKNN2 layout for board-side offline Chinese TTS:

- melotts_rknn.py: upstream TTS inference entrypoint.
- encoder.onnx: CPU encoder model loaded by onnxruntime.
- decoder.rknn: RKNN decoder model loaded by rknn-toolkit-lite2.
- wheels/: offline Python wheelhouse bundled into the runtime package.
- pydeps/: board-side Python dependencies installed locally by tools/install_python_deps.sh.
- english_utils/ and text/: upstream text normalization helpers.
- lexicon.txt and tokens.txt: pronunciation assets used by the frontend.
- g.bin: static speaker embedding blob expected by the upstream script.

Board-side prerequisites before running the smoke test:

- python3 must be available in PATH.
- tools/install_python_deps.sh will install the bundled offline wheelhouse into ./pydeps automatically during upload/all.

Typical demo invocation:

    ./run_tts.sh "短波电台指令测试。"

Current acceleration boundary in this bundle:

- encoder: CPU/ONNX.
- decoder: RKNN/NPU.

Licensing note:

- The upstream MeloTTS-RKNN2 mirror is published under AGPL-3.0.
- Treat this bundle as a technical integration path first; complete legal review before promoting it to the default commercial delivery line.