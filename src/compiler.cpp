#include "compiler.hpp"

#include <slang-rhi.h>

std::expected<Slang::ComPtr<slang::IComponentType>, mazorca::error_code> 
mazorca::compile_shader(const std::filesystem::path& shader_file_path, Slang::ComPtr<slang::IGlobalSession>& globalSession) {

    std::ifstream shader_file(shader_file_path, std::ios::binary);

    if (!shader_file) {
        std::println("[{}] [ERROR] Unable to read shader file: {}", current_time(), shader_file_path.string());
        return std::unexpected(mazorca::error_code::invalid);
    }

    // Read shader source to string for Slang shader compiler
    std::string shader_source{
        std::istreambuf_iterator<char>(shader_file), 
        std::istreambuf_iterator<char>()
    };    

    // List of enabled compilation targets
    slang::TargetDesc targetDesc = {
        .format = SLANG_SPIRV,
        .profile = globalSession->findProfile("spirv_1_6")
    };

    // Create session
    // Note: create path to parent directory due to lifetime of C string
    std::filesystem::path module_search_path {shader_file_path.parent_path()};
    std::array<const char*, 1> searchPaths = {module_search_path.c_str()};
    slang::SessionDesc sessionDesc = {
        .targets = &targetDesc,
        .targetCount = 1,
        .searchPaths = searchPaths.data(),
        .searchPathCount = searchPaths.size()
    };

    // Create the session
    Slang::ComPtr<slang::ISession> session;
    globalSession->createSession(sessionDesc, session.writeRef());

    // Load single module
    // TODO: if multiple modules, each should be loaded in order of imports
    // For now, import other Slang source files into a single translational unit
    Slang::ComPtr<slang::IModule> slangModule;
    {
        Slang::ComPtr<slang::IBlob> diagnosticsBlob;
        
        slangModule = session->loadModule(
            shader_file_path.stem().c_str(),    // Module name
            diagnosticsBlob.writeRef()          // Optional diagnostic container
        );
        if (diagnosticsBlob != nullptr) {
            std::println("{}", static_cast<const char*>(diagnosticsBlob->getBufferPointer()));
        }
        if (!slangModule) {
            return std::unexpected(mazorca::error_code::invalid);
        }
    }

    // Query entry points and compose module
    int num_entry_points = slangModule->getDefinedEntryPointCount();
    std::vector<slang::IComponentType*> componentTypes;
    for (int i = 0; i < num_entry_points; i++) {
        Slang::ComPtr<slang::IEntryPoint> entryPoint;
        slangModule->getDefinedEntryPoint(i, entryPoint.writeRef());
        componentTypes.emplace_back(entryPoint.get());
    }

    Slang::ComPtr<slang::IComponentType> composedProgram;
    {
        Slang::ComPtr<slang::IBlob> diagnosticsBlob;
        SlangResult result = session->createCompositeComponentType(
            componentTypes.data(),
            static_cast<SlangInt>(componentTypes.size()),
            composedProgram.writeRef(),
            diagnosticsBlob.writeRef());
        if (diagnosticsBlob != nullptr) {
            std::println("{}", static_cast<const char*>(diagnosticsBlob->getBufferPointer()));
        }
        if (SLANG_FAILED(result))
            return std::unexpected(mazorca::error_code::invalid);
    }

    // Linking
    Slang::ComPtr<slang::IComponentType> linkedProgram;
    {
        Slang::ComPtr<slang::IBlob> diagnosticsBlob;
        SlangResult result = composedProgram->link(
            linkedProgram.writeRef(),
            diagnosticsBlob.writeRef());
        if (diagnosticsBlob != nullptr) {
            std::println("{}", static_cast<const char*>(diagnosticsBlob->getBufferPointer()));
        }
        if (SLANG_FAILED(result))
            return std::unexpected(mazorca::error_code::invalid);
    }

    // Perform reflection on compiled and linked program layout
    slang::ProgramLayout* programLayout = linkedProgram->getLayout();
    for (std::size_t i = 0; i < programLayout->getEntryPointCount(); i++) {
        Slang::ComPtr<slang::IBlob> diagnosticsBlob;
        Slang::ComPtr<slang::IBlob> spirvBlob;
        SlangResult result = linkedProgram->getEntryPointCode(
            static_cast<SlangInt>(i),   // Entry point index
            0,                          // Target index
            spirvBlob.writeRef(),
            diagnosticsBlob.writeRef());
        if (diagnosticsBlob != nullptr) {
            std::println("{} [ERROR] {}", mazorca::current_time(), static_cast<const char*>(diagnosticsBlob->getBufferPointer()));
        }
        if (SLANG_FAILED(result)) {
            std::println("{} [ERROR] Failed to obtain entry points from linked Slang program.", mazorca::current_time());
            return std::unexpected(mazorca::error_code::invalid);
        }
        std::println("{} bytes compiled for entry point {}.", spirvBlob->getBufferSize(), programLayout->getEntryPointByIndex(i)->getName());
    }

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

    rhi::ShaderProgramDesc shader_program_desc {
        .slangGlobalScope = linkedProgram
    };
    
    Slang::ComPtr<rhi::IShaderProgram> shaderProgram {device->createShaderProgram(shader_program_desc)};

    rhi::ComputePipelineDesc compute_pipeline_desc {
        .program = shaderProgram
    };

    // Slang::ComPtr<rhi::IComputePipeline> compute_pipeline {device->createComputePipeline(compute_pipeline_desc)};

    return linkedProgram;
}
