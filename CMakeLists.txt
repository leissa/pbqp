cmake_minimum_required(VERSION 3.25)

project(pbqp VERSION 0.1.0 LANGUAGES CXX)

option(PBQP_BUILD_TESTS "Build the unit tests" ON)
option(PBQP_BUILD_EXAMPLE "Build the demo executable" ON)
option(PBQP_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
add_library(pbqp
    src/pbqp/graph.cpp
    src/pbqp/solver.cpp
)
add_library(pbqp::pbqp ALIAS pbqp)

find_package(Threads REQUIRED)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/submodules/svector/include/ankerl/svector.h")
    message(FATAL_ERROR "submodules are missing; run `git submodule update --init`")
endif()

target_include_directories(pbqp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(pbqp SYSTEM PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/submodules/svector/include>
)
target_compile_features(pbqp PUBLIC cxx_std_23)
target_link_libraries(pbqp PUBLIC Threads::Threads)
set_target_properties(pbqp PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

if(MSVC)
    target_compile_options(pbqp PRIVATE /W4)
    if(PBQP_WARNINGS_AS_ERRORS)
        target_compile_options(pbqp PRIVATE /WX)
    endif()
else()
    target_compile_options(pbqp PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    if(PBQP_WARNINGS_AS_ERRORS)
        target_compile_options(pbqp PRIVATE -Werror)
    endif()
endif()

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if(PBQP_BUILD_EXAMPLE)
    add_executable(pbqp_demo apps/pbqp_demo.cpp)
    target_link_libraries(pbqp_demo PRIVATE pbqp::pbqp)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(PBQP_BUILD_TESTS)
    enable_testing()

    add_executable(pbqp_tests tests/test_pbqp.cpp)
    target_link_libraries(pbqp_tests PRIVATE pbqp::pbqp)

    add_test(NAME pbqp_tests COMMAND pbqp_tests)
endif()
