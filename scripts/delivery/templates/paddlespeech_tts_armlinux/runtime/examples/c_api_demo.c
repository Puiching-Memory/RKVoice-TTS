#include <stdio.h>
#include "rkvoice_tts_api.h"

int main(void) {
    rkvoice_tts_config_t config;
    rkvoice_tts_config_init(&config);
    config.front_conf_path = "./front.conf";
    config.acoustic_model_path = "./models/cpu/fastspeech2_csmsc_arm.nb";
    config.vocoder_model_path = "./models/cpu/mb_melgan_csmsc_arm.nb";

    rkvoice_tts_engine_t* engine = NULL;
    rkvoice_tts_status_t status = rkvoice_tts_create(&config, &engine);
    if (status != RKVOICE_TTS_STATUS_OK) {
        fprintf(stderr, "create failed: %s\n", rkvoice_tts_status_string(status));
        return 1;
    }

    rkvoice_tts_metrics_t metrics;
    status = rkvoice_tts_synthesize_to_file(
        engine,
        "短波电台指令测试。",
        "./output/c_api_demo.wav",
        &metrics);
    if (status != RKVOICE_TTS_STATUS_OK) {
        fprintf(stderr, "synthesize failed: %s, %s\n",
                rkvoice_tts_status_string(status),
                rkvoice_tts_last_error(engine));
        rkvoice_tts_destroy(engine);
        return 1;
    }

    printf("inference=%.3f ms, wav_size=%d bytes, wav_duration=%.3f ms, rtf=%.6f\n",
           metrics.inference_time_ms,
           metrics.wav_size_bytes,
           metrics.wav_duration_ms,
           metrics.rtf);
    rkvoice_tts_destroy(engine);
    return 0;
}
