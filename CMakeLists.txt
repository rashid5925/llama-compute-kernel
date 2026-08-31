cmake_minimum_required(VERSION 4.3)
project(VulkanTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Vulkan REQUIRED)

add_executable(llama-compute-kernel llama_compute.cpp)

target_link_libraries(llama-compute-kernel Vulkan::Vulkan)
