cmake_minimum_required(VERSION 3.10)
project(rkvoice_tts_sdk)

option(WITH_FRONT_DEMO "Build front demo" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(ABSL_PROPAGATE_CXX_STD ON)

set(CMAKE_SYSTEM_NAME Linux)
if(ARM_ABI STREQUAL "armv8")
    set(CMAKE_SYSTEM_PROCESSOR aarch64)
elseif(ARM_ABI STREQUAL "armv7hf")
    set(CMAKE_SYSTEM_PROCESSOR arm)
else()
    message(FATAL_ERROR "Unknown arch abi ${ARM_ABI}, only support armv8 and armv7hf.")
    return()
endif()

message(STATUS "TARGET ARCH ABI: ${ARM_ABI}")
message(STATUS "PADDLE LITE DIR: ${PADDLE_LITE_DIR}")

include_directories(${PADDLE_LITE_DIR}/include)
link_directories(${PADDLE_LITE_DIR}/libs/${ARM_ABI})
link_directories(${PADDLE_LITE_DIR}/lib)

if(ARM_ABI STREQUAL "armv8")
    set(CMAKE_CXX_FLAGS "-march=armv8-a ${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "-march=armv8-a ${CMAKE_C_FLAGS}")
elseif(ARM_ABI STREQUAL "armv7hf")
    set(CMAKE_CXX_FLAGS "-march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4 ${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "-march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4 ${CMAKE_C_FLAGS}")
endif()

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND OR OpenMP_CXX_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "Found OpenMP ${OpenMP_VERSION} ${OpenMP_CXX_VERSION}")
else()
    message(FATAL_ERROR "Could not found OpenMP!")
    return()
endif()

add_subdirectory(TTSCppFrontend)

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}
    TTSCppFrontend/src
    TTSCppFrontend/third-party/vendor/cppjieba/include
    TTSCppFrontend/third-party/vendor/limonp/include
)

add_library(rkvoice_tts_shared SHARED rkvoice_tts_core.cc)
set_target_properties(rkvoice_tts_shared PROPERTIES OUTPUT_NAME rkvoice_tts)
target_link_libraries(rkvoice_tts_shared paddle_light_api_shared paddlespeech_tts_front)

add_library(rkvoice_tts_static STATIC rkvoice_tts_core.cc)
set_target_properties(rkvoice_tts_static PROPERTIES OUTPUT_NAME rkvoice_tts)
target_link_libraries(rkvoice_tts_static paddle_light_api_shared paddlespeech_tts_front)

add_executable(rkvoice_tts_demo rkvoice_tts_demo_main.cc)
target_link_libraries(rkvoice_tts_demo rkvoice_tts_shared)

add_executable(paddlespeech_tts_demo rkvoice_tts_demo_main.cc)
target_link_libraries(paddlespeech_tts_demo rkvoice_tts_shared)
