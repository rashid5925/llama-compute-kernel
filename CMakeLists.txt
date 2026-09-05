cmake_minimum_required(VERSION 4.3)
project(llama-compute-kernel)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/O2) # MSVC equivalent of -O3
else()
    add_compile_options($<$<CONFIG:Release>:-O3>)
endif()

find_package(Vulkan REQUIRED)

set(SHADER_SOURCES
    shaders/rmsnorm.comp
    shaders/matmul.comp
    shaders/rope.comp
    shaders/attention.comp
    shaders/swiglu.comp
    shaders/residualadd.comp
)

set(SPIRV_BINARY_FILES "")
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    set(SPV_FILE "${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME}.spv")

    add_custom_command(
        OUTPUT ${SPV_FILE}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/shaders"
        COMMAND Vulkan::glslc -O ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER} -o ${SPV_FILE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}
        COMMENT "Compiling GLSL shader ${SHADER} -> ${SPV_FILE}"
    )

    list(APPEND SPIRV_BINARY_FILES ${SPV_FILE})
endforeach()

add_custom_target(CompileShaders DEPENDS ${SPIRV_BINARY_FILES})

add_executable(llama-compute-kernel llama_compute.cpp)

target_link_libraries(llama-compute-kernel Vulkan::Vulkan)

add_dependencies(llama-compute-kernel CompileShaders)
