cmake_minimum_required(VERSION 3.20)
project(minemind LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

option(MINEMIND_BUILD_DASHBOARD "Build the SDL3 + Dear ImGui dashboard" OFF)

find_package(OpenMP REQUIRED)
find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Sim (headless) -> build/_craftsim.<abi>.so, imported by craftsim.py.
set(CRAFTSIM_SOURCES src/bindings.cpp src/sim.cpp src/mobs.cpp src/survival.cpp src/nether.cpp src/build.cpp)
pybind11_add_module(_craftsim ${CRAFTSIM_SOURCES})
target_include_directories(_craftsim PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(_craftsim PRIVATE cxx_std_20)
target_compile_options(_craftsim PRIVATE -Wall -Wextra)
target_link_libraries(_craftsim PRIVATE OpenMP::OpenMP_CXX)
set_target_properties(_craftsim PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

if(EXISTS "${CMAKE_SOURCE_DIR}/dashboard/CMakeLists.txt" AND MINEMIND_BUILD_DASHBOARD)
    add_subdirectory(dashboard)
endif()
