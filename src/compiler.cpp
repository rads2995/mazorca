#include "compiler.hpp"

#include <slang-rhi.h>

std::expected<void, mazorca::error_code> 
mazorca::compile_shader(mazorca::vulkan_data& vulkan_data, const std::filesystem::path& shader_file_path, Slang::ComPtr<slang::IGlobalSession>& globalSession) {

    std::ifstream shader_file(shader_file_path, std::ios::binary);

    if (!shader_file) {
        std::println("[{}] [ERROR] Unable to read shader file: {}", current_time(), shader_file_path.string());
        return std::unexpected(mazorca::error_code::invalid);
    }

    // Read shader source to string for Slang shader compiler
    std::string shader_source {
        std::istreambuf_iterator<char>(shader_file), 
        std::istreambuf_iterator<char>()
    };    

    // List of enabled compilation targets
    slang::TargetDesc targetDesc {
        .format = SLANG_SPIRV,
        .profile = globalSession->findProfile("spirv_1_6")
    };

    // Create session descriptor
    // Note: create new local path object from shader path due to lifetime of c_str
    std::filesystem::path module_search_path {shader_file_path.parent_path()};
    std::array<const char*, 1> searchPaths {module_search_path.c_str()};
    slang::SessionDesc sessionDesc {
        .targets = &targetDesc,
        .targetCount = 1,
        .searchPaths = searchPaths.data(),
        .searchPathCount = searchPaths.size()
    };

    // Create local session and add it to the global session
    // Note: local session must live for as long as we need the compiled shader program
    Slang::ComPtr<slang::ISession> session;
    if(SLANG_FAILED(globalSession->createSession(sessionDesc, session.writeRef()))) {
        std::println("[{}] [ERROR] Failed to create Slang local session.", mazorca::current_time());
        return std::unexpected(mazorca::error_code::invalid);
    }

    // Load Slang module from provided input file path
    // For now, we import other Slang source files into a single translational unit
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
            std::println("[{}] [ERROR] Failed to load Slang module into local session.", current_time());
            return std::unexpected(mazorca::error_code::invalid);
        }
    }

    // Identify number and names of entry points inside Slang module
    SlangInt32 num_entry_points = slangModule->getDefinedEntryPointCount();
    std::vector<Slang::ComPtr<slang::IEntryPoint>> entryPoints;
    entryPoints.reserve(static_cast<std::size_t>(num_entry_points));
    for (SlangInt32 i = 0; i < num_entry_points; i++) {
        Slang::ComPtr<slang::IEntryPoint> entryPoint;
        if(SLANG_FAILED(slangModule->getDefinedEntryPoint(i, entryPoint.writeRef()))) {
            std::println("[{}] [ERROR] Failed to find entry point at index: {}", mazorca::current_time(), i);
            return std::unexpected(mazorca::error_code::invalid);
        }
        entryPoints.emplace_back(std::move(entryPoint));
    }

    // Compose Slang module
    // Note: the number of components is the number of loaded Slang modules plus the number of entry points
    std::vector<slang::IComponentType*> componentTypes;
    SlangInt num_loaded_modules = session->getLoadedModuleCount();
    componentTypes.reserve(static_cast<std::size_t>(num_loaded_modules) + static_cast<std::size_t>(num_entry_points));
    // Get the loaded Slang modules for this local session
    // Note: we want pointers here, the lifetimes are managed by the templated ComPtr objects
    // Also, if modules have some for of dependency, the order in which they are loaded is important!
    for (SlangInt i = 0; i < num_loaded_modules; i++) {
        componentTypes.emplace_back(session->getLoadedModule(i));
    }   
    // Get the loaded entry points for this local session
    for (const auto& entry_point : entryPoints) {
        componentTypes.emplace_back(entry_point);
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
        if (SLANG_FAILED(result)) {
            std::println("[{}] [ERROR] Failed to create composed Slang program.", mazorca::current_time());
            return std::unexpected(mazorca::error_code::invalid);
        }
    }

    // Link the composed Slang program
    Slang::ComPtr<slang::IComponentType> linkedProgram;
    {
        Slang::ComPtr<slang::IBlob> diagnosticsBlob;
        SlangResult result = composedProgram->link(
            linkedProgram.writeRef(),
            diagnosticsBlob.writeRef());
        if (diagnosticsBlob != nullptr) {
            std::println("{}", static_cast<const char*>(diagnosticsBlob->getBufferPointer()));
        }
        if (SLANG_FAILED(result)) {
            std::println("[{}] [ERROR] Failed to link the composed Slang program.", mazorca::current_time());   
            return std::unexpected(mazorca::error_code::invalid);
        }
    }

    // Running the composed and linked Slang program on device
    // TODO: debug everything below this point, as it causes segfaults!
    rhi::DeviceDesc deviceDesc {
        .deviceType = rhi::DeviceType::Vulkan,
        .slang {
            .slangGlobalSession = globalSession.get(),
            .targetProfile = "spirv_1_6"
        },
        .enableValidation = true
    };

    Slang::ComPtr<rhi::IDevice> device;
    {
        SlangResult result = rhi::getRHI()->createDevice(deviceDesc, device.writeRef());
        if (SLANG_FAILED(result)) {
            std::println("[{}] [ERROR] Failed to create Slang RHI device object.", mazorca::current_time());   
            return std::unexpected(mazorca::error_code::invalid);
        }
    }

    rhi::ShaderProgramDesc shader_program_desc {
        .slangGlobalScope = linkedProgram.get()
    };

    Slang::ComPtr<rhi::IShaderProgram> shaderProgram;
    {
        Slang::ComPtr<slang::IBlob> diagnosticsBlob;
        SlangResult result = device->createShaderProgram(
            shader_program_desc,
            shaderProgram.writeRef(),
            diagnosticsBlob.writeRef()
        );
        if (diagnosticsBlob != nullptr) {
            std::println("{}", static_cast<const char*>(diagnosticsBlob->getBufferPointer()));
        }
        if (SLANG_FAILED(result)) {
            std::println("[{}] [ERROR] Failed to create Slang shader program.", mazorca::current_time());   
            return std::unexpected(mazorca::error_code::invalid);
        }
    }

    rhi::ComputePipelineDesc compute_pipeline_desc {
        .program = shaderProgram.get()
    };

    Slang::ComPtr<rhi::IComputePipeline> compute_pipeline;
    {
        SlangResult result = device->createComputePipeline(
            compute_pipeline_desc,
            compute_pipeline.writeRef()
        );
        if (SLANG_FAILED(result)) {
            std::println("[{}] [ERROR] Failed to create Slang compute pipeline.", mazorca::current_time());   
            return std::unexpected(mazorca::error_code::invalid);
        }
    }

    return {};
}
