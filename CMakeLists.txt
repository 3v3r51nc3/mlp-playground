cmake_minimum_required(VERSION 3.20)
project(mlp-playground LANGUAGES CXX)

# --- Configuration ---

# C++17 code (Release config) is compatible with C++20.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include the 'include' directory
include_directories(include)

# --- Source Files ---

set(SOURCES
    main.cpp
    src/layer.cpp
    src/mnist_loader.cpp
    src/vector_utils.cpp
    src/matrix.cpp
    src/neural_network.cpp
)

set(HEADERS
    include/layer.h
    include/matrix.h
    include/mnist_loader.h
    include/neural_network.h
    include/vector_utils.h
)

# --- Build Executable ---

# Create the executable with sources and headers
# (Headers are added so they appear in IDE project trees like VS)
add_executable(mlp-playground ${SOURCES} ${HEADERS})

# --- OpenMP Support ---

# Enabled OpenMP for Release builds. 
# We will look for it and link it if found.
find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found. Linking...")
    target_link_libraries(mlp-playground PRIVATE OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found. Building without parallelism.")
endif()

# --- Compiler Options ---

if(MSVC)
    # Match Warning Level 3 from vcxproj
    target_compile_options(mlp-playground PRIVATE /W3)
    
    # Ensure standard compliance options if strictly needed (optional)
    target_compile_options(mlp-playground PRIVATE /permissive-)
else()
    # Equivalent warnings for GCC/Clang (optional, just for portability)
    target_compile_options(mlp-playground PRIVATE -Wall -Wextra)
endif()