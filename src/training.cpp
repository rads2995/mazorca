#include "training.hpp"

#include <slang-rhi.h>

std::expected<void, mazorca::error_code> 
mazorca::train_model(Slang::ComPtr<slang::IComponentType>& slang_program, Slang::ComPtr<slang::IGlobalSession>& globalSession) {

    rhi::DeviceDesc deviceDesc {
        .deviceType = rhi::DeviceType::Vulkan,
        .slang {
            .slangGlobalSession = globalSession,
            .targetProfile = "spirv_1_6"
        },
    };

    Slang::ComPtr<rhi::IDevice> device {rhi::getRHI()->createDevice(deviceDesc)};
    if (!device) {
        std::println("Failed to create Slang RHI device object!");
        return std::unexpected(mazorca::error_code::invalid);
    }

    std::println("{}", device->getInfo().apiName);

    rhi::ShaderProgramDesc shader_program_desc {
        .slangGlobalScope = slang_program
    };
    
    // Slang::ComPtr<rhi::IShaderProgram> shaderProgram {device->createShaderProgram(shader_program_desc)};

    // rhi::ComputePipelineDesc compute_pipeline_desc {
    //     .program = shaderProgram
    // };

    // Slang::ComPtr<rhi::IComputePipeline> compute_pipeline {device->createComputePipeline(compute_pipeline_desc)};

    return {};
}
