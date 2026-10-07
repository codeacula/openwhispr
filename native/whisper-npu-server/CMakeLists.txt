cmake_minimum_required(VERSION 3.20)
project(whisper_npu_server LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT WIN32)
  message(FATAL_ERROR "whisper-npu-server targets Windows on Snapdragon only")
endif()

# Fetched and pinned by scripts/build-whisper-npu-server.js
foreach(var ORT_INCLUDE_DIR THIRD_PARTY_INCLUDE_DIR)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "${var} must be set (run scripts/build-whisper-npu-server.js)")
  endif()
endforeach()

add_executable(
  whisper-npu-server
  src/main.cpp
  src/log_mel.cpp
  src/ort_runtime.cpp
  src/vocab.cpp
  src/wav.cpp
  src/whisper_npu.cpp
)

target_include_directories(
  whisper-npu-server
  PRIVATE
  src
  ${ORT_INCLUDE_DIR}
  ${THIRD_PARTY_INCLUDE_DIR}
)

# onnxruntime.dll is loaded by full path at startup (see ort_runtime.cpp), so
# nothing links its import library and the loader never searches for it.
target_compile_definitions(
  whisper-npu-server
  PRIVATE
  ORT_API_MANUAL_INIT
  NOMINMAX
  WIN32_LEAN_AND_MEAN
  UNICODE
  _UNICODE
)

target_compile_options(whisper-npu-server PRIVATE /W4 /utf-8 /permissive-)

# Static CRT for our own code; the bundled onnxruntime.dll brings its own
# MSVC runtime DLLs, which the pack ships beside it.
set_property(TARGET whisper-npu-server PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
