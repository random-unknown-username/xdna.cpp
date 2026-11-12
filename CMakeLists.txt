cmake_minimum_required(VERSION 3.20)

project(xdna.cpp VERSION 1.1.0 LANGUAGES CXX)

option(XDNA_ENABLE_XRT "Build the XRT-backed discovery probe when XRT is available" ON)
set(XDNA_XRT_PKG_CONFIG_DIR "/opt/xilinx/xrt/lib/pkgconfig" CACHE PATH
    "Directory containing xrt.pc")

# The llama.cpp plugin and the tools that exercise it need the ggml headers from
# a llama.cpp checkout. Point this at one with -DXDNA_LLAMA_CPP_DIR=... or set
# LLAMA_CPP_DIR. Everything else builds without it.
set(XDNA_LLAMA_CPP_DIR "" CACHE PATH
    "Path to a llama.cpp checkout (ggml headers plus the plugin ABI)")
if (NOT XDNA_LLAMA_CPP_DIR AND DEFINED ENV{LLAMA_CPP_DIR})
    set(XDNA_LLAMA_CPP_DIR "$ENV{LLAMA_CPP_DIR}")
endif()
if (NOT XDNA_LLAMA_CPP_DIR)
    foreach(candidate "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp"
                     "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp")
        if (EXISTS "${candidate}/ggml/include/ggml.h")
            set(XDNA_LLAMA_CPP_DIR "${candidate}")
            break()
        endif()
    endforeach()
endif()
set(XDNA_GGML_HEADERS_FOUND OFF)
if (XDNA_LLAMA_CPP_DIR AND EXISTS "${XDNA_LLAMA_CPP_DIR}/ggml/include/ggml.h")
    set(XDNA_GGML_HEADERS_FOUND ON)
    set(XDNA_GGML_INCLUDE_DIRS
        "${XDNA_LLAMA_CPP_DIR}/ggml/include"
        "${XDNA_LLAMA_CPP_DIR}/ggml/src")
    message(STATUS "xdna.cpp: ggml headers from ${XDNA_LLAMA_CPP_DIR}")
else()
    message(STATUS "xdna.cpp: no llama.cpp checkout found; skipping the ggml plugin")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)
include(CTest)

find_package(OpenMP REQUIRED)

add_library(xdna-core STATIC
    src/xdna-device.cpp
    src/xdna-gate1.cpp
    src/xdna-kernel-probe.cpp
    src/xdna-q4-0-reference.cpp
    src/cpu-q4-gemv-engine.cpp
    src/xdna-q4-prepack.cpp
    src/xdna-gemv-engine.cpp
    src/dma_streamer.cpp
)
add_library(xdna::core ALIAS xdna-core)

target_include_directories(xdna-core
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_compile_features(xdna-core PUBLIC cxx_std_17)
target_link_libraries(xdna-core PUBLIC OpenMP::OpenMP_CXX)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(xdna-core PRIVATE -Wall -Wextra -Wpedantic -fPIC -mavx2 -mfma -mavx512f -mavx512bw -mavx512vl -mavx512vnni)
endif()

add_library(xdna-kernel INTERFACE)
add_library(xdna::kernel ALIAS xdna-kernel)
target_link_libraries(xdna-kernel INTERFACE xdna::core)

set(XDNA_HAS_XRT OFF)
if (XDNA_ENABLE_XRT)
    find_package(PkgConfig QUIET)
    if (PkgConfig_FOUND)
        # The configured path is prepended only for this CMake process. An
        # existing caller-supplied PKG_CONFIG_PATH is retained.
        if (IS_DIRECTORY "${XDNA_XRT_PKG_CONFIG_DIR}")
            if (DEFINED ENV{PKG_CONFIG_PATH} AND NOT "$ENV{PKG_CONFIG_PATH}" STREQUAL "")
                set(ENV{PKG_CONFIG_PATH}
                    "${XDNA_XRT_PKG_CONFIG_DIR}:$ENV{PKG_CONFIG_PATH}")
            else()
                set(ENV{PKG_CONFIG_PATH} "${XDNA_XRT_PKG_CONFIG_DIR}")
            endif()
        endif()

        pkg_check_modules(XRT QUIET IMPORTED_TARGET xrt)
        if (XRT_FOUND)
            set(XDNA_HAS_XRT ON)
            message(STATUS "xdna.cpp: XRT support enabled (${XRT_VERSION})")
        else()
            message(STATUS "xdna.cpp: XRT not found; using the no-XRT probe")
        endif()
    else()
        message(STATUS "xdna.cpp: pkg-config not found; using the no-XRT probe")
    endif()
else()
    message(STATUS "xdna.cpp: XRT support disabled by XDNA_ENABLE_XRT=OFF")
endif()

if (XDNA_HAS_XRT)
    target_link_libraries(xdna-core PUBLIC PkgConfig::XRT uuid)
    target_compile_definitions(xdna-core PUBLIC XDNA_WITH_XRT=1)
    if (XRT_LIBRARY_DIRS)
        set_property(TARGET xdna-core APPEND PROPERTY BUILD_RPATH
            "${XRT_LIBRARY_DIRS}")
    endif()
endif()

if (XDNA_HAS_XRT)
    set(XDNA_XRT_PROBE_SOURCE src/xdna-xrt-probe.cpp)
else()
    set(XDNA_XRT_PROBE_SOURCE src/xdna-xrt-probe-stub.cpp)
endif()

add_library(xdna-probe STATIC ${XDNA_XRT_PROBE_SOURCE})
add_library(xdna::probe ALIAS xdna-probe)
target_include_directories(xdna-probe
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(xdna-probe PUBLIC xdna::core)
target_compile_features(xdna-probe PUBLIC cxx_std_17)

if (XDNA_HAS_XRT)
    target_link_libraries(xdna-probe PRIVATE PkgConfig::XRT)
    target_compile_definitions(xdna-probe PRIVATE XDNA_WITH_XRT=1)
    if (XRT_LIBRARY_DIRS)
        set_property(TARGET xdna-probe APPEND PROPERTY BUILD_RPATH
            "${XRT_LIBRARY_DIRS}")
    endif()
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(xdna-probe PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(xdna-info tools/xdna-info.cpp)
target_link_libraries(xdna-info PRIVATE xdna::kernel xdna::probe)
target_compile_features(xdna-info PRIVATE cxx_std_17)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(xdna-info PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(xdna-gate1 tools/xdna-gate1.cpp)
target_link_libraries(xdna-gate1 PRIVATE xdna::core)
target_compile_features(xdna-gate1 PRIVATE cxx_std_17)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(xdna-gate1 PRIVATE -Wall -Wextra -Wpedantic)
endif()
if (XDNA_HAS_XRT)
    target_link_libraries(xdna-gate1 PRIVATE PkgConfig::XRT)
    target_compile_definitions(xdna-gate1 PRIVATE XDNA_WITH_XRT=1)
    if (XRT_LIBRARY_DIRS)
        set_property(TARGET xdna-gate1 APPEND PROPERTY BUILD_RPATH
            "${XRT_LIBRARY_DIRS}")
    endif()
endif()

add_executable(xdna-gguf-info tools/xdna-gguf-info.cpp)
target_compile_features(xdna-gguf-info PRIVATE cxx_std_17)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(xdna-gguf-info PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(xdna-bench tools/xdna-bench.cpp)
target_link_libraries(xdna-bench PRIVATE xdna::core)
target_compile_features(xdna-bench PRIVATE cxx_std_17)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(xdna-bench PRIVATE -Wall -Wextra -Wpedantic)
endif()
if (XDNA_HAS_XRT)
    target_link_libraries(xdna-bench PRIVATE PkgConfig::XRT)
    target_compile_definitions(xdna-bench PRIVATE XDNA_WITH_XRT=1)
    if (XRT_LIBRARY_DIRS)
        set_property(TARGET xdna-bench APPEND PROPERTY BUILD_RPATH
            "${XRT_LIBRARY_DIRS}")
    endif()
endif()

add_executable(xdna-kernel-bench tools/xdna-kernel-bench.cpp)
target_link_libraries(xdna-kernel-bench PRIVATE xdna::core OpenMP::OpenMP_CXX)
target_compile_features(xdna-kernel-bench PRIVATE cxx_std_17)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(xdna-kernel-bench PRIVATE -Wall -Wextra -Wpedantic -mavx2 -mfma -mavx512f -mavx512bw -mavx512vl -mavx512vnni)
endif()
if (XDNA_HAS_XRT)
    target_link_libraries(xdna-kernel-bench PRIVATE PkgConfig::XRT)
    target_compile_definitions(xdna-kernel-bench PRIVATE XDNA_WITH_XRT=1)
    if (XRT_LIBRARY_DIRS)
        set_property(TARGET xdna-kernel-bench APPEND PROPERTY BUILD_RPATH
            "${XRT_LIBRARY_DIRS}")
    endif()
endif()

if (BUILD_TESTING)
    add_executable(xdna-test-device tests/test-device.cpp)
    target_link_libraries(xdna-test-device PRIVATE xdna::core)
    target_compile_features(xdna-test-device PRIVATE cxx_std_17)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(xdna-test-device PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME xdna-device-parsing COMMAND xdna-test-device)

    add_executable(xdna-test-kernel-probe tests/test-kernel-probe.cpp)
    target_link_libraries(xdna-test-kernel-probe PRIVATE xdna::kernel)
    target_compile_features(xdna-test-kernel-probe PRIVATE cxx_std_17)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(xdna-test-kernel-probe PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME xdna-kernel-probe-formatting COMMAND xdna-test-kernel-probe)

    add_executable(xdna-test-xrt-probe tests/test-xrt-probe.cpp)
    target_link_libraries(xdna-test-xrt-probe PRIVATE xdna::core)
    target_compile_features(xdna-test-xrt-probe PRIVATE cxx_std_17)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(xdna-test-xrt-probe PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME xdna-xrt-probe-formatting COMMAND xdna-test-xrt-probe)

    add_executable(xdna-test-gate1 tests/test-gate1.cpp)
    target_link_libraries(xdna-test-gate1 PRIVATE xdna::core)
    target_compile_features(xdna-test-gate1 PRIVATE cxx_std_17)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(xdna-test-gate1 PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME xdna-gate1-validation COMMAND xdna-test-gate1)

    add_executable(xdna-test-q4-0-reference tests/test-q4-0-reference.cpp)
    target_link_libraries(xdna-test-q4-0-reference PRIVATE xdna::core)
    target_compile_features(xdna-test-q4-0-reference PRIVATE cxx_std_17)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(xdna-test-q4-0-reference PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME xdna-q4-0-reference COMMAND xdna-test-q4-0-reference)

    if (XDNA_GGML_HEADERS_FOUND)
        add_executable(xdna-test-backend tests/test-backend.cpp)
        target_link_libraries(xdna-test-backend PRIVATE xdna::core ggml-xdna)
        target_compile_features(xdna-test-backend PRIVATE cxx_std_17)
        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(xdna-test-backend PRIVATE -Wall -Wextra -Wpedantic)
        endif()
        add_test(NAME xdna-backend-test COMMAND xdna-test-backend)
    endif()

    add_executable(xdna-test-q4-prepack tests/test-q4-prepack.cpp)
    target_link_libraries(xdna-test-q4-prepack PRIVATE xdna::core)
    target_compile_features(xdna-test-q4-prepack PRIVATE cxx_std_17)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(xdna-test-q4-prepack PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME xdna-q4-prepack COMMAND xdna-test-q4-prepack)

    add_executable(xdna-test-dma-streaming tests/test-dma-streaming.cpp)
    target_link_libraries(xdna-test-dma-streaming PRIVATE xdna::core)
    target_compile_features(xdna-test-dma-streaming PRIVATE cxx_std_17)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(xdna-test-dma-streaming PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    if (XDNA_HAS_XRT)
        target_link_libraries(xdna-test-dma-streaming PRIVATE PkgConfig::XRT)
        target_compile_definitions(xdna-test-dma-streaming PRIVATE XDNA_WITH_XRT=1)
        if (XRT_LIBRARY_DIRS)
            set_property(TARGET xdna-test-dma-streaming APPEND PROPERTY BUILD_RPATH
                "${XRT_LIBRARY_DIRS}")
        endif()
    endif()
    add_test(NAME xdna-dma-streaming COMMAND xdna-test-dma-streaming)

    add_executable(xdna-test-gemv-accuracy tests/test-gemv-accuracy.cpp)
    target_link_libraries(xdna-test-gemv-accuracy PRIVATE xdna::core)
    target_compile_features(xdna-test-gemv-accuracy PRIVATE cxx_std_17)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(xdna-test-gemv-accuracy PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    if (XDNA_HAS_XRT)
        target_link_libraries(xdna-test-gemv-accuracy PRIVATE PkgConfig::XRT)
        target_compile_definitions(xdna-test-gemv-accuracy PRIVATE XDNA_WITH_XRT=1)
        if (XRT_LIBRARY_DIRS)
            set_property(TARGET xdna-test-gemv-accuracy APPEND PROPERTY BUILD_RPATH
                "${XRT_LIBRARY_DIRS}")
        endif()
    endif()
    add_test(NAME xdna-gemv-accuracy COMMAND xdna-test-gemv-accuracy)
endif()

set(XDNA_INSTALL_TARGETS
    xdna-info xdna-gate1 xdna-bench xdna-gguf-info xdna-kernel-bench)

if (XDNA_GGML_HEADERS_FOUND)
    add_library(ggml-xdna SHARED src/ggml-xdna.cpp)
    target_link_libraries(ggml-xdna PRIVATE xdna::core)
    target_include_directories(ggml-xdna PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        ${XDNA_GGML_INCLUDE_DIRS}
    )
    target_compile_features(ggml-xdna PUBLIC cxx_std_17)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(ggml-xdna PRIVATE -Wall -Wextra -Wpedantic -fPIC -mavx2 -mfma -mavx512f -mavx512bw -mavx512vl -mavx512vnni)
    endif()
    if (XDNA_HAS_XRT)
        target_link_libraries(ggml-xdna PRIVATE PkgConfig::XRT)
        target_compile_definitions(ggml-xdna PRIVATE XDNA_WITH_XRT=1)
        if (XRT_LIBRARY_DIRS)
            set_property(TARGET ggml-xdna APPEND PROPERTY BUILD_RPATH "${XRT_LIBRARY_DIRS}")
        endif()
    endif()

    if (BUILD_TESTING)
        add_executable(xdna-test-kquants-gemv tests/test-kquants-gemv.cpp)
        target_link_libraries(xdna-test-kquants-gemv PRIVATE xdna::core ggml-xdna OpenMP::OpenMP_CXX)
        target_include_directories(xdna-test-kquants-gemv PUBLIC
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
            ${XDNA_GGML_INCLUDE_DIRS})
        target_compile_features(xdna-test-kquants-gemv PRIVATE cxx_std_17)
        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(xdna-test-kquants-gemv PRIVATE -Wall -Wextra -Wpedantic -mavx2 -mfma -mavx512f -mavx512bw -mavx512vl -mavx512vnni)
        endif()
        if (XDNA_HAS_XRT)
            target_link_libraries(xdna-test-kquants-gemv PRIVATE PkgConfig::XRT)
            target_compile_definitions(xdna-test-kquants-gemv PRIVATE XDNA_WITH_XRT=1)
            if (XRT_LIBRARY_DIRS)
                set_property(TARGET xdna-test-kquants-gemv APPEND PROPERTY BUILD_RPATH
                    "${XRT_LIBRARY_DIRS}")
            endif()
        endif()
        add_test(NAME xdna-kquants-gemv COMMAND xdna-test-kquants-gemv)
    endif()

    add_executable(xdna-bandwidth-bench tools/xdna-bandwidth-bench.cpp)
    target_link_libraries(xdna-bandwidth-bench PRIVATE
        xdna::core
        ggml-xdna
        OpenMP::OpenMP_CXX
    )
    target_include_directories(xdna-bandwidth-bench PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        ${XDNA_GGML_INCLUDE_DIRS})
    target_compile_features(xdna-bandwidth-bench PRIVATE cxx_std_17)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(xdna-bandwidth-bench PRIVATE -Wall -Wextra -Wpedantic -mavx2 -mfma -mavx512f -mavx512bw -mavx512vl -mavx512vnni)
    endif()
    if (XDNA_HAS_XRT)
        target_link_libraries(xdna-bandwidth-bench PRIVATE PkgConfig::XRT)
        target_compile_definitions(xdna-bandwidth-bench PRIVATE XDNA_WITH_XRT=1)
        if (XRT_LIBRARY_DIRS)
            set_property(TARGET xdna-bandwidth-bench APPEND PROPERTY BUILD_RPATH "${XRT_LIBRARY_DIRS}")
        endif()
    endif()

    list(APPEND XDNA_INSTALL_TARGETS xdna-bandwidth-bench ggml-xdna)
endif()

install(TARGETS ${XDNA_INSTALL_TARGETS}
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
