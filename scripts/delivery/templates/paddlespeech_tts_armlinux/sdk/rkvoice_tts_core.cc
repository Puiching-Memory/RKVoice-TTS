#include "rkvoice_tts_api.h"

#include <front/front_interface.h>
#include <glog/logging.h>
#include <limonp/StringUtil.hpp>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "Predictor.hpp"

using namespace paddle::lite_api;

namespace {

constexpr char kDefaultFrontConf[] = "./front.conf";
constexpr char kDefaultCpuAcousticModel[] = "./models/cpu/fastspeech2_csmsc_arm.nb";
constexpr char kDefaultCpuVocoder[] = "./models/cpu/mb_melgan_csmsc_arm.nb";
constexpr char kDefaultSentence[] = "你好，欢迎使用语音合成服务";

std::string ResolvePath(const char* value, const char* fallback) {
    if (value != nullptr && value[0] != '\0') {
        return value;
    }
    return fallback;
}

class RkVoiceTtsEngineImpl {
  public:
    explicit RkVoiceTtsEngineImpl(const rkvoice_tts_config_t& raw_config)
        : backend_(raw_config.backend),
          front_conf_path_(ResolvePath(raw_config.front_conf_path, kDefaultFrontConf)),
          acoustic_model_path_(ResolvePath(raw_config.acoustic_model_path, kDefaultCpuAcousticModel)),
          vocoder_model_path_(ResolvePath(raw_config.vocoder_model_path, kDefaultCpuVocoder)),
          wav_sample_rate_(raw_config.wav_sample_rate == 0 ? 24000 : raw_config.wav_sample_rate),
          cpu_thread_num_(raw_config.cpu_thread_num <= 0 ? 1 : raw_config.cpu_thread_num),
          wav_bit_depth_(raw_config.wav_bit_depth == RKVOICE_TTS_WAV_BIT_DEPTH_FLOAT32
                             ? RKVOICE_TTS_WAV_BIT_DEPTH_FLOAT32
                             : RKVOICE_TTS_WAV_BIT_DEPTH_PCM16) {}

    rkvoice_tts_status_t Init() {
        if (backend_ == RKVOICE_TTS_BACKEND_RKNN) {
            return Fail(
                RKVOICE_TTS_STATUS_UNSUPPORTED_BACKEND,
                "RKNN backend is reserved for the NPU delivery path, but is not compiled into this runtime bundle yet.");
        }
        if (backend_ != RKVOICE_TTS_BACKEND_CPU) {
            return Fail(RKVOICE_TTS_STATUS_INVALID_ARGUMENT, "Unknown TTS backend.");
        }

        front_inst_ = std::make_unique<ppspeech::FrontEngineInterface>(front_conf_path_);
        if (!front_inst_ || front_inst_->init()) {
            front_inst_.reset();
            return Fail(RKVOICE_TTS_STATUS_INIT_FAILED, "Create tts front engine failed.");
        }

        if (wav_bit_depth_ == RKVOICE_TTS_WAV_BIT_DEPTH_PCM16) {
            predictor_.reset(new Predictor<int16_t>());
        } else if (wav_bit_depth_ == RKVOICE_TTS_WAV_BIT_DEPTH_FLOAT32) {
            predictor_.reset(new Predictor<float>());
        } else {
            return Fail(RKVOICE_TTS_STATUS_INVALID_ARGUMENT, "Unsupported WAV bit depth.");
        }

        if (!predictor_->Init(
                acoustic_model_path_,
                vocoder_model_path_,
                PowerMode::LITE_POWER_HIGH,
                cpu_thread_num_,
                wav_sample_rate_)) {
            predictor_.reset();
            return Fail(RKVOICE_TTS_STATUS_INIT_FAILED, "Predictor init failed.");
        }
        return RKVOICE_TTS_STATUS_OK;
    }

    const std::string& last_error() const { return last_error_; }

    rkvoice_tts_status_t SynthesizeToFile(
        const char* sentence_utf8,
        const char* output_wav_path,
        rkvoice_tts_metrics_t* out_metrics) {
        if (!front_inst_ || !predictor_) {
            return Fail(RKVOICE_TTS_STATUS_INIT_FAILED, "Engine is not initialized.");
        }
        std::string sentence = ResolvePath(sentence_utf8, kDefaultSentence);
        std::string output_path = ResolvePath(output_wav_path, "./output/tts.wav");

        std::vector<int64_t> phones;
        rkvoice_tts_status_t status = SentenceToPhones(sentence, &phones);
        if (status != RKVOICE_TTS_STATUS_OK) {
            return status;
        }

        if (!predictor_->RunModel(phones)) {
            return Fail(RKVOICE_TTS_STATUS_SYNTH_FAILED, "Predictor run model failed.");
        }

        if (!predictor_->WriteWavToFile(output_path)) {
            return Fail(RKVOICE_TTS_STATUS_IO_ERROR, "Write wav file failed.");
        }

        if (out_metrics != nullptr) {
            out_metrics->inference_time_ms = predictor_->GetInferenceTime();
            out_metrics->wav_size_bytes = predictor_->GetWavSize();
            out_metrics->wav_duration_ms = predictor_->GetWavDuration();
            out_metrics->rtf = predictor_->GetRTF();
        }
        last_error_.clear();
        return RKVOICE_TTS_STATUS_OK;
    }

  private:
    rkvoice_tts_status_t SentenceToPhones(
        const std::string& sentence_utf8,
        std::vector<int64_t>* phones_out) {
        if (phones_out == nullptr) {
            return Fail(RKVOICE_TTS_STATUS_INVALID_ARGUMENT, "Output phone buffer is null.");
        }

        std::wstring ws_sentence = ppspeech::utf8string2wstring(sentence_utf8);

        std::wstring sentence_simp;
        front_inst_->Trand2Simp(ws_sentence, &sentence_simp);
        ws_sentence = sentence_simp;

        std::string s_sentence;
        std::vector<std::wstring> sentence_part;
        std::vector<int> phoneids;
        std::vector<int> toneids;

        LOG(INFO) << "Start to segment sentences by punctuation";
        front_inst_->SplitByPunc(ws_sentence, &sentence_part);
        LOG(INFO) << "Segment sentences through punctuation successfully";

        LOG(INFO) << "Start to get the phoneme and tone id sequence of each sentence";
        for (int index = 0; index < static_cast<int>(sentence_part.size()); ++index) {
            LOG(INFO) << "Raw sentence is: "
                      << ppspeech::wstring2utf8string(sentence_part[index]);
            front_inst_->SentenceNormalize(&sentence_part[index]);
            s_sentence = ppspeech::wstring2utf8string(sentence_part[index]);
            LOG(INFO) << "After normalization sentence is: " << s_sentence;

            if (0 != front_inst_->GetSentenceIds(s_sentence, &phoneids, &toneids)) {
                return Fail(
                    RKVOICE_TTS_STATUS_SYNTH_FAILED,
                    "Front engine failed to generate phone ids and tone ids.");
            }
        }

        LOG(INFO) << "The phoneids of the sentence is: "
                  << limonp::Join(phoneids.begin(), phoneids.end(), " ");
        LOG(INFO) << "The toneids of the sentence is: "
                  << limonp::Join(toneids.begin(), toneids.end(), " ");
        LOG(INFO) << "Get the phoneme id sequence of each sentence successfully";

        phones_out->resize(phoneids.size());
        std::transform(phoneids.begin(), phoneids.end(), phones_out->begin(), [](int value) {
            return static_cast<int64_t>(value);
        });
        return RKVOICE_TTS_STATUS_OK;
    }

    rkvoice_tts_status_t Fail(rkvoice_tts_status_t status, const std::string& message) {
        last_error_ = message;
        LOG(ERROR) << message;
        return status;
    }

  private:
    rkvoice_tts_backend_t backend_;
    std::string front_conf_path_;
    std::string acoustic_model_path_;
    std::string vocoder_model_path_;
    uint32_t wav_sample_rate_;
    int cpu_thread_num_;
    rkvoice_tts_wav_bit_depth_t wav_bit_depth_;
    std::string last_error_;
    std::unique_ptr<ppspeech::FrontEngineInterface> front_inst_;
    std::unique_ptr<PredictorInterface> predictor_;
};

}  // namespace

struct rkvoice_tts_engine {
    std::unique_ptr<RkVoiceTtsEngineImpl> impl;
};

extern "C" {

void rkvoice_tts_config_init(rkvoice_tts_config_t* config) {
    if (config == nullptr) {
        return;
    }
    config->backend = RKVOICE_TTS_BACKEND_CPU;
    config->front_conf_path = kDefaultFrontConf;
    config->acoustic_model_path = kDefaultCpuAcousticModel;
    config->vocoder_model_path = kDefaultCpuVocoder;
    config->wav_sample_rate = 24000;
    config->cpu_thread_num = 1;
    config->wav_bit_depth = RKVOICE_TTS_WAV_BIT_DEPTH_PCM16;
}

int rkvoice_tts_backend_is_available(rkvoice_tts_backend_t backend) {
    return backend == RKVOICE_TTS_BACKEND_CPU ? 1 : 0;
}

const char* rkvoice_tts_status_string(rkvoice_tts_status_t status) {
    switch (status) {
        case RKVOICE_TTS_STATUS_OK:
            return "ok";
        case RKVOICE_TTS_STATUS_INVALID_ARGUMENT:
            return "invalid_argument";
        case RKVOICE_TTS_STATUS_INIT_FAILED:
            return "init_failed";
        case RKVOICE_TTS_STATUS_SYNTH_FAILED:
            return "synth_failed";
        case RKVOICE_TTS_STATUS_IO_ERROR:
            return "io_error";
        case RKVOICE_TTS_STATUS_UNSUPPORTED_BACKEND:
            return "unsupported_backend";
        default:
            return "unknown";
    }
}

rkvoice_tts_status_t rkvoice_tts_create(
    const rkvoice_tts_config_t* config,
    rkvoice_tts_engine_t** out_engine) {
    if (config == nullptr || out_engine == nullptr) {
        return RKVOICE_TTS_STATUS_INVALID_ARGUMENT;
    }
    *out_engine = nullptr;

    std::unique_ptr<rkvoice_tts_engine_t> engine(new rkvoice_tts_engine_t());
    engine->impl = std::make_unique<RkVoiceTtsEngineImpl>(*config);
    rkvoice_tts_status_t status = engine->impl->Init();
    if (status != RKVOICE_TTS_STATUS_OK) {
        return status;
    }

    *out_engine = engine.release();
    return RKVOICE_TTS_STATUS_OK;
}

void rkvoice_tts_destroy(rkvoice_tts_engine_t* engine) {
    delete engine;
}

const char* rkvoice_tts_last_error(const rkvoice_tts_engine_t* engine) {
    if (engine == nullptr || !engine->impl) {
        return "";
    }
    return engine->impl->last_error().c_str();
}

rkvoice_tts_status_t rkvoice_tts_synthesize_to_file(
    rkvoice_tts_engine_t* engine,
    const char* sentence_utf8,
    const char* output_wav_path,
    rkvoice_tts_metrics_t* out_metrics) {
    if (engine == nullptr || !engine->impl) {
        return RKVOICE_TTS_STATUS_INVALID_ARGUMENT;
    }
    return engine->impl->SynthesizeToFile(sentence_utf8, output_wav_path, out_metrics);
}

}  // extern "C"
