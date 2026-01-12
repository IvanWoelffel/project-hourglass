cmake_minimum_required(VERSION 3.15)

project(ProjectHourglass)

# --------------------------------------------------
# C++ standard
# --------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --------------------------------------------------
# SDL
# --------------------------------------------------

option(SDL_SHARED "" OFF)
option(SDL_STATIC "" ON)
add_subdirectory(libs/SDL-release-3.2.24)

# --------------------------------------------------
# geGL
# --------------------------------------------------

add_subdirectory(libs/geGL-master)

# --------------------------------------------------
# Assimp configuration
# --------------------------------------------------

set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(external/assimp)

# --------------------------------------------------
# Executable
# --------------------------------------------------

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/hourglass.cpp
)

# --------------------------------------------------
# Include directories
# --------------------------------------------------

target_include_directories(${PROJECT_NAME} PRIVATE
    src/
    ${PROJECT_SOURCE_DIR}/include
)

# --------------------------------------------------
# Link libraries
# --------------------------------------------------

target_link_libraries(${PROJECT_NAME} PRIVATE
    SDL3::SDL3
    geGL::geGL
    assimp
)

# --------------------------------------------------
# Runtime assets
# --------------------------------------------------

file(COPY
    ${PROJECT_SOURCE_DIR}/models
    DESTINATION ${CMAKE_BINARY_DIR}
)

# --------------------------------------------------
# Convenience target
# --------------------------------------------------

add_custom_target(run ./${PROJECT_NAME})
