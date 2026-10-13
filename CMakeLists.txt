cmake_minimum_required(VERSION 3.22)
project(CrunchyPy VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Verify submodules
# ---------------------------------------------------------------------------
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/modules/JUCE/CMakeLists.txt")
    message(FATAL_ERROR "JUCE submodule is missing. Please run:\n  git submodule update --init --recursive")
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/modules/RTNeural/CMakeLists.txt")
    message(FATAL_ERROR "RTNeural submodule is missing. Please run:\n  git submodule update --init --recursive")
endif()

add_subdirectory(modules/JUCE)

# ---------------------------------------------------------------------------
# RTNeural backend selection
#
# Backend choice is delegated to RTNeural's cmake/ChooseBackend.cmake.
# xsimd with AVX2 provides 256-bit SIMD vector execution for both the
# GRU neural networks and dense layers at 48 kHz.
# ---------------------------------------------------------------------------
set(CRUNCHYPY_BACKEND "xsimd" CACHE STRING "SIMD backend for RTNeural: xsimd, eigen or stl")
set_property(CACHE CRUNCHYPY_BACKEND PROPERTY STRINGS xsimd eigen stl)

if(CRUNCHYPY_BACKEND STREQUAL "xsimd")
    set(RTNEURAL_XSIMD ON  CACHE BOOL "" FORCE)
    set(RTNEURAL_EIGEN OFF CACHE BOOL "" FORCE)
    set(RTNEURAL_STL   OFF CACHE BOOL "" FORCE)
elseif(CRUNCHYPY_BACKEND STREQUAL "eigen")
    set(RTNEURAL_XSIMD OFF CACHE BOOL "" FORCE)
    set(RTNEURAL_EIGEN ON  CACHE BOOL "" FORCE)
    set(RTNEURAL_STL   OFF CACHE BOOL "" FORCE)
else()
    set(RTNEURAL_XSIMD OFF CACHE BOOL "" FORCE)
    set(RTNEURAL_EIGEN OFF CACHE BOOL "" FORCE)
    set(RTNEURAL_STL   ON  CACHE BOOL "" FORCE)
endif()

add_subdirectory(modules/RTNeural)

if(CRUNCHYPY_BACKEND STREQUAL "stl")
    message(WARNING "CrunchyPy -- RTNeural backend: scalar STL. Not real-time safe for this network.")
endif()

# ---------------------------------------------------------------------------
# SIMD Vectorization & Alignment Flags
# ---------------------------------------------------------------------------
if(MSVC)
    target_compile_options(RTNeural PUBLIC /arch:AVX2)
    target_compile_definitions(RTNeural PUBLIC RTNEURAL_AVX_ENABLED=1 RTNEURAL_DEFAULT_ALIGNMENT=32)
    message(STATUS "CrunchyPy -- AVX2 enabled on RTNeural (MSVC)")
elseif(APPLE)
    message(STATUS "CrunchyPy -- macOS build configured")
else()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        target_compile_options(RTNeural PUBLIC -mavx2 -mfma)
        target_compile_definitions(RTNeural PUBLIC RTNEURAL_AVX_ENABLED=1 RTNEURAL_DEFAULT_ALIGNMENT=32)
    endif()
    message(STATUS "CrunchyPy -- Linux build configured")
endif()

# ---------------------------------------------------------------------------
# Plugin Declaration (VST3 & Standalone, AU on macOS)
# ---------------------------------------------------------------------------
if(APPLE)
    set(CRUNCHYPY_FORMATS VST3 AU Standalone)
else()
    set(CRUNCHYPY_FORMATS VST3 Standalone)
endif()

juce_add_plugin(CrunchyPy
    COMPANY_NAME "Yahia Kemari"
    BUNDLE_ID com.yahiakemari.crunchypy
    PLUGIN_MANUFACTURER_CODE YhKm
    PLUGIN_CODE MbfB
    FORMATS ${CRUNCHYPY_FORMATS}
    ProductName "CrunchyPy"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD FALSE
)

target_compile_definitions(CrunchyPy PUBLIC
    JUCE_DISPLAY_SPLASH_SCREEN=0
    JUCE_REPORT_APP_USAGE=0
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_STANDALONE_OWN_WRAPPER=1
    JUCE_MODAL_LOOPS_PERMITTED=0
    JUCE_MODULE_DEFINITION_NO_CPP11_APIS=0
    JUCE_ENABLE_COREAUDIO_FONT_SCALING=0
)

target_sources(CrunchyPy PRIVATE
    src/Plugin.cpp
    src/Editor.cpp
    src/EQStage.cpp
)

# ---------------------------------------------------------------------------
# Embedded Neural Network Models
#
# preamp.json: Mesa Boogie Mark IIC+ Preamp (13 features, 96 hidden GRU, 12 conditioned knobs)
# poweramp.json: Mesa Boogie Mark IIC+ 100W Poweramp (2 features, 32 hidden GRU, Presence conditioned)
# ---------------------------------------------------------------------------
juce_add_binary_data(BinaryData SOURCES
    models/preamp.json
    models/poweramp.json
)
set_target_properties(BinaryData PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

target_link_libraries(CrunchyPy PUBLIC
    BinaryData
    juce::juce_audio_utils
    juce::juce_dsp
    juce::juce_gui_extra
    RTNeural
)

target_include_directories(CrunchyPy PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/modules/RTNeural/RTNeural"
)

juce_generate_juce_header(CrunchyPy)

# ---------------------------------------------------------------------------
# Bundle Repair
#
# JUCE 8.0.12 writes the VST3 bundle's moduleinfo.json with trailing commas,
# which breaks strict JSON parsers in modern DAWs (e.g., Bitwig, Reaper).
# FixModuleInfoJson normalises it to strict standard JSON.
# ---------------------------------------------------------------------------
add_custom_target(CrunchyPyFixBundle ALL
    COMMAND ${CMAKE_COMMAND}
            "-DSTART=$<TARGET_FILE_DIR:CrunchyPy_VST3>"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FixModuleInfoJson.cmake"
    DEPENDS CrunchyPy_VST3
    COMMENT "Normalising moduleinfo.json to strict JSON"
    VERBATIM
)

# ---------------------------------------------------------------------------
# Release Packaging (Windows)
# ---------------------------------------------------------------------------
if(WIN32)
    set(crunchypyPackageDir "${CMAKE_CURRENT_SOURCE_DIR}/dist")

    add_custom_target(CrunchyPyPackage ALL
        COMMAND ${CMAKE_COMMAND}
                "-DSTAGE=$<TARGET_FILE_DIR:CrunchyPy>"
                "-DDEST=${crunchypyPackageDir}"
                "-DREADMESRC=${CMAKE_CURRENT_SOURCE_DIR}/packaging/README.txt"
                "-DBUNDLE=${PROJECT_NAME}.vst3"
                "-DEXE=${PROJECT_NAME}.exe"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Package.cmake"
        DEPENDS CrunchyPyFixBundle CrunchyPy_Standalone
        COMMENT "Packaging CrunchyPy into dist/"
        VERBATIM
    )
endif()
