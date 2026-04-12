cmake_minimum_required(VERSION 3.10)
project(tts_third_party_libs)

include(ExternalProject)

set(VENDOR_DIR ${CMAKE_CURRENT_LIST_DIR}/vendor)
set(COMMON_CROSS_CMAKE_ARGS
    -DCMAKE_SYSTEM_NAME=Linux
    -DCMAKE_SYSTEM_PROCESSOR=aarch64
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
)

macro(assert_vendor_dir name)
    if(NOT EXISTS ${VENDOR_DIR}/${name})
        message(FATAL_ERROR "Missing vendored dependency: ${VENDOR_DIR}/${name}")
    endif()
endmacro()

assert_vendor_dir(gflags)
assert_vendor_dir(glog)
assert_vendor_dir(abseil-cpp)
assert_vendor_dir(cppjieba)
assert_vendor_dir(limonp)

ExternalProject_Add(gflags
    SOURCE_DIR      ${VENDOR_DIR}/gflags
    PREFIX          ${CMAKE_CURRENT_BINARY_DIR}
    INSTALL_DIR     ${CMAKE_CURRENT_BINARY_DIR}
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND   ""
    CMAKE_ARGS
        ${COMMON_CROSS_CMAKE_ARGS}
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DBUILD_STATIC_LIBS=OFF
        -DBUILD_SHARED_LIBS=ON
        -DBUILD_TESTING=OFF
)

ExternalProject_Add(glog
    SOURCE_DIR      ${VENDOR_DIR}/glog
    PREFIX          ${CMAKE_CURRENT_BINARY_DIR}
    INSTALL_DIR     ${CMAKE_CURRENT_BINARY_DIR}
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND   ""
    CMAKE_ARGS
        ${COMMON_CROSS_CMAKE_ARGS}
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DBUILD_TESTING=OFF
    DEPENDS         gflags
)

ExternalProject_Add(abseil
    SOURCE_DIR      ${VENDOR_DIR}/abseil-cpp
    PREFIX          ${CMAKE_CURRENT_BINARY_DIR}
    INSTALL_DIR     ${CMAKE_CURRENT_BINARY_DIR}
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND   ""
    CMAKE_ARGS
        ${COMMON_CROSS_CMAKE_ARGS}
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DABSL_PROPAGATE_CXX_STD=ON
        -DABSL_ENABLE_INSTALL=ON
        -DBUILD_TESTING=OFF
        -DABSL_BUILD_TESTING=OFF
)

ExternalProject_Add(cppjieba
    SOURCE_DIR       ${VENDOR_DIR}/cppjieba
    PREFIX           ${CMAKE_CURRENT_BINARY_DIR}
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND   ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
)

ExternalProject_Add(limonp
    SOURCE_DIR       ${VENDOR_DIR}/limonp
    PREFIX           ${CMAKE_CURRENT_BINARY_DIR}
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND   ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
)
