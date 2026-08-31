cmake_minimum_required(VERSION 3.20)

project(xdna.cpp VERSION 0.1.0 LANGUAGES CXX)

option(XDNA_ENABLE_XRT "Build the XRT-backed discovery probe when XRT is available" ON)
set(XDNA_XRT_PKG_CONFIG_DIR "/opt/xilinx/xrt/lib/pkgconfig" CACHE PATH
    "Directory containing xrt.pc")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)
include(CTest)

add_library(xdna-core STATIC
    src/xdna-device.cpp
    src/xdna-gate1.cpp
    src/xdna-kernel-probe.cpp
    src/xdna-q4-0-reference.cpp
    src/cpu-q4-gemv-engine.cpp
    src/xdna-q4-prepack.cpp
    src/xdna-gemv-engine.cpp
)
add_library(xdna::core ALIAS xdna-core)

target_include_directories(xdna-core
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_compile_features(xdna-core PUBLIC cxx_std_17)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(xdna-core PRIVATE -Wall -Wextra -Wpedantic -fPIC -mavx2 -mfma)
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
target_link_libraries(xdna-kernel-bench PRIVATE xdna::core)
target_compile_features(xdna-kernel-bench PRIVATE cxx_std_17)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(xdna-kernel-bench PRIVATE -Wall -Wextra -Wpedantic -mavx2 -mfma)
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

    add_executable(xdna-test-backend tests/test-backend.cpp)
    target_link_libraries(xdna-test-backend PRIVATE xdna::core ggml-xdna)
    target_compile_features(xdna-test-backend PRIVATE cxx_std_17)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(xdna-test-backend PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME xdna-backend-test COMMAND xdna-test-backend)

    add_executable(xdna-test-q4-prepack tests/test-q4-prepack.cpp)
    target_link_libraries(xdna-test-q4-prepack PRIVATE xdna::core)
    target_compile_features(xdna-test-q4-prepack PRIVATE cxx_std_17)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(xdna-test-q4-prepack PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME xdna-q4-prepack COMMAND xdna-test-q4-prepack)
endif()

add_library(ggml-xdna SHARED
    src/ggml-xdna.cpp
)
target_link_libraries(ggml-xdna PRIVATE xdna::core)
target_include_directories(ggml-xdna PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    "/home/satvik/FORK/llama.cpp/ggml/include"
    "/home/satvik/FORK/llama.cpp/ggml/src"
)
target_compile_features(ggml-xdna PUBLIC cxx_std_17)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ggml-xdna PRIVATE -Wall -Wextra -Wpedantic -fPIC)
endif()
if (XDNA_HAS_XRT)
    target_link_libraries(ggml-xdna PRIVATE PkgConfig::XRT)
    target_compile_definitions(ggml-xdna PRIVATE XDNA_WITH_XRT=1)
    if (XRT_LIBRARY_DIRS)
        set_property(TARGET ggml-xdna APPEND PROPERTY BUILD_RPATH "${XRT_LIBRARY_DIRS}")
    endif()
endif()

add_executable(xdna-cli tools/xdna-cli.cpp)
target_link_libraries(xdna-cli PRIVATE xdna::core ggml-xdna)
target_include_directories(xdna-cli PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "/home/satvik/FORK/llama.cpp/ggml/include"
    "/home/satvik/FORK/llama.cpp/ggml/src"
)
target_compile_features(xdna-cli PRIVATE cxx_std_17)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(xdna-cli PRIVATE -Wall -Wextra -Wpedantic -mavx2 -mfma)
endif()
if (XDNA_HAS_XRT)
    target_link_libraries(xdna-cli PRIVATE PkgConfig::XRT)
    target_compile_definitions(xdna-cli PRIVATE XDNA_WITH_XRT=1)
    if (XRT_LIBRARY_DIRS)
        set_property(TARGET xdna-cli APPEND PROPERTY BUILD_RPATH "${XRT_LIBRARY_DIRS}")
    endif()
endif()

install(TARGETS xdna-info xdna-gate1 xdna-bench xdna-gguf-info xdna-kernel-bench xdna-cli ggml-xdna
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
