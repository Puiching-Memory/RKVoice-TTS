#ifndef RKVOICE_TTS_API_H_
#define RKVOICE_TTS_API_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    RKVOICE_TTS_STATUS_OK = 0,
    RKVOICE_TTS_STATUS_INVALID_ARGUMENT = 1,
    RKVOICE_TTS_STATUS_INIT_FAILED = 2,
    RKVOICE_TTS_STATUS_SYNTH_FAILED = 3,
    RKVOICE_TTS_STATUS_IO_ERROR = 4,
    RKVOICE_TTS_STATUS_UNSUPPORTED_BACKEND = 5,
} rkvoice_tts_status_t;

typedef enum {
    RKVOICE_TTS_BACKEND_CPU = 0,
    RKVOICE_TTS_BACKEND_RKNN = 1,
} rkvoice_tts_backend_t;

typedef enum {
    RKVOICE_TTS_WAV_BIT_DEPTH_PCM16 = 16,
    RKVOICE_TTS_WAV_BIT_DEPTH_FLOAT32 = 32,
} rkvoice_tts_wav_bit_depth_t;

typedef struct {
    rkvoice_tts_backend_t backend;
    const char* front_conf_path;
    const char* acoustic_model_path;
    const char* vocoder_model_path;
    uint32_t wav_sample_rate;
    int cpu_thread_num;
    rkvoice_tts_wav_bit_depth_t wav_bit_depth;
} rkvoice_tts_config_t;

typedef struct {
    float inference_time_ms;
    int wav_size_bytes;
    float wav_duration_ms;
    float rtf;
} rkvoice_tts_metrics_t;

typedef struct rkvoice_tts_engine rkvoice_tts_engine_t;

void rkvoice_tts_config_init(rkvoice_tts_config_t* config);
int rkvoice_tts_backend_is_available(rkvoice_tts_backend_t backend);
const char* rkvoice_tts_status_string(rkvoice_tts_status_t status);
rkvoice_tts_status_t rkvoice_tts_create(
    const rkvoice_tts_config_t* config,
    rkvoice_tts_engine_t** out_engine);
void rkvoice_tts_destroy(rkvoice_tts_engine_t* engine);
const char* rkvoice_tts_last_error(const rkvoice_tts_engine_t* engine);
rkvoice_tts_status_t rkvoice_tts_synthesize_to_file(
    rkvoice_tts_engine_t* engine,
    const char* sentence_utf8,
    const char* output_wav_path,
    rkvoice_tts_metrics_t* out_metrics);

#ifdef __cplusplus
}
#endif

#endif
