#include <gflags/gflags.h>
#include <glog/logging.h>
#include <cstdlib>
#include <string>

#include "rkvoice_tts_api.h"

DEFINE_string(
    sentence,
    "你好，欢迎使用语音合成服务",
    "Text to be synthesized (Chinese only. English will crash the program.)");
DEFINE_string(front_conf, "./front.conf", "Front configuration file");
DEFINE_string(acoustic_model,
              "./models/cpu/fastspeech2_csmsc_arm.nb",
              "Acoustic model file. Current runtime bundle ships the CPU .nb model.");
DEFINE_string(vocoder,
              "./models/cpu/mb_melgan_csmsc_arm.nb",
              "Vocoder model file. Current runtime bundle ships the CPU .nb model.");
DEFINE_string(output_wav, "./output/tts.wav", "Output WAV file");
DEFINE_string(wav_bit_depth,
              "16",
              "WAV bit depth, 16 (16-bit PCM) or 32 (32-bit IEEE float)");
DEFINE_string(wav_sample_rate,
              "24000",
              "WAV sample rate, should match the output of the vocoder");
DEFINE_string(cpu_thread, "1", "CPU thread numbers");
DEFINE_string(backend, "cpu", "Backend name: cpu or rknn");

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    rkvoice_tts_config_t config;
    rkvoice_tts_config_init(&config);
    config.front_conf_path = FLAGS_front_conf.c_str();
    config.acoustic_model_path = FLAGS_acoustic_model.c_str();
    config.vocoder_model_path = FLAGS_vocoder.c_str();
    config.wav_sample_rate = static_cast<uint32_t>(std::strtoul(FLAGS_wav_sample_rate.c_str(), nullptr, 10));
    config.cpu_thread_num = std::atoi(FLAGS_cpu_thread.c_str());

    if (FLAGS_wav_bit_depth == "16") {
        config.wav_bit_depth = RKVOICE_TTS_WAV_BIT_DEPTH_PCM16;
    } else if (FLAGS_wav_bit_depth == "32") {
        config.wav_bit_depth = RKVOICE_TTS_WAV_BIT_DEPTH_FLOAT32;
    } else {
        LOG(ERROR) << "Unsupported WAV bit depth: " << FLAGS_wav_bit_depth;
        return -1;
    }

    if (FLAGS_backend == "cpu") {
        config.backend = RKVOICE_TTS_BACKEND_CPU;
    } else if (FLAGS_backend == "rknn") {
        config.backend = RKVOICE_TTS_BACKEND_RKNN;
    } else {
        LOG(ERROR) << "Unsupported backend: " << FLAGS_backend;
        return -1;
    }

    rkvoice_tts_engine_t* engine = nullptr;
    rkvoice_tts_status_t status = rkvoice_tts_create(&config, &engine);
    if (status != RKVOICE_TTS_STATUS_OK) {
        LOG(ERROR) << "rkvoice_tts_create failed: " << rkvoice_tts_status_string(status);
        return -1;
    }

    rkvoice_tts_metrics_t metrics = {};
    status = rkvoice_tts_synthesize_to_file(
        engine,
        FLAGS_sentence.c_str(),
        FLAGS_output_wav.c_str(),
        &metrics);
    if (status != RKVOICE_TTS_STATUS_OK) {
        LOG(ERROR) << "rkvoice_tts_synthesize_to_file failed: "
                   << rkvoice_tts_status_string(status) << ", "
                   << rkvoice_tts_last_error(engine);
        rkvoice_tts_destroy(engine);
        return -1;
    }

    LOG(INFO) << "Inference time: " << metrics.inference_time_ms << " ms, "
              << "WAV size (without header): " << metrics.wav_size_bytes
              << " bytes, "
              << "WAV duration: " << metrics.wav_duration_ms << " ms, "
              << "RTF: " << metrics.rtf;

    rkvoice_tts_destroy(engine);
    return 0;
}
