#include "training.hpp"
#include "compiler.hpp"

#include <slang-rhi.h>

struct Kernel {

    Slang::ComPtr<rhi::IShaderProgram> program;
    Slang::ComPtr<rhi::IComputePipeline> pipeline;
    operator bool() {
        return program && pipeline;
    }
};

struct NetworkParameterAllocation {
    
    std::size_t weightsOffset;
    std::size_t weightsSize;
    std::size_t biasOffset;
    std::size_t biasSize;
    std::size_t weightsGradOffset;
    std::size_t biasGradOffset;
};

std::expected<std::unordered_map<std::string, Slang::ComPtr<slang::IBlob>>, mazorca::error_code> 
mazorca::train_model(const std::filesystem::path& shader_file_path, Slang::ComPtr<slang::IGlobalSession>& globalSession) {

    Slang::ComPtr<rhi::IDevice> gDevice;

    Slang::ComPtr<slang::ISession> gSlangSession;
    Slang::ComPtr<slang::IModule> gSlangModule;

    Kernel gLearnGradProgram;
    Kernel gAdjustParamProgram;

    rhi::DeviceDesc deviceDesc {
        .slang.targetProfile = "spirv_1_6",
        .deviceType = rhi::DeviceType::Vulkan
    };

    gDevice = rhi::getRHI()->createDevice(deviceDesc);
    if (!gDevice) {
        return std::unexpected(mazorca::error_code::invalid);
    }

    gSlangSession = gDevice->getSlangSession();
    
    // TODO: compile shader here!

    // auto program = gDevice->createShaderProgram(linkedProgram);
    // rhi::ComputePipelineDesc desc {
    //     .program = program.get();
        
    // }
    
    // Kernel result;
    // result.program = program;
    // result.pipeline = gDevice->createComputePipeline(desc)

}
