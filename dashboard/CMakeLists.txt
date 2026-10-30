# MineMind Training Studio (C++20, SDL3 + OpenGL 3.3 + Dear ImGui docking).
# Standalone:  cmake -S dashboard -B dashboard/build && cmake --build dashboard/build
# Also usable as a subdirectory of the root project.
cmake_minimum_required(VERSION 3.24)
project(minemind_studio LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(SDL3 REQUIRED CONFIG)
find_package(nlohmann_json 3.11 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)

set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui)
if(NOT EXISTS ${IMGUI_DIR}/imgui.h)
    message(FATAL_ERROR
        "Dear ImGui (docking) missing. Run:\n"
        "  git clone --depth 1 -b docking https://github.com/ocornut/imgui dashboard/third_party/imgui")
endif()

# Third-party code: built with its own warning level.
add_library(imgui_mm STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdl3.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp)
target_include_directories(imgui_mm PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
target_link_libraries(imgui_mm PUBLIC SDL3::SDL3 OpenGL::GL)
if(MSVC)
    target_compile_options(imgui_mm PRIVATE /w)
else()
    target_compile_options(imgui_mm PRIVATE -w)
endif()

add_executable(minemind_studio
    src/main.cpp
    src/gl_procs.cpp
    src/ui_util.cpp
    src/panel_viewport.cpp
    src/panel_training.cpp
    src/panel_controls.cpp
    src/panel_agent.cpp
    src/panel_graphs.cpp
    src/panel_nn.cpp
    src/panel_checkpoints.cpp
    src/panel_console.cpp
    src/panel_chat.cpp
    net/client.cpp
    net/protocol.cpp)
target_include_directories(minemind_studio PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/net
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_link_libraries(minemind_studio PRIVATE
    imgui_mm
    nlohmann_json::nlohmann_json
    Threads::Threads)
target_compile_options(minemind_studio PRIVATE -Wall -Wextra -Wno-unused-parameter)
