#include "compiler.hpp"

#include <slang-rhi.h>

std::expected<void, mazorca::error_code> 
mazorca::compile_shader(const std::filesystem::path& shader_file_path, const Slang::ComPtr<slang::IGlobalSession>& globalSession) {

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
    // Note: create new local path object of file parent path due to lifetime of c_str
    std::filesystem::path module_search_path {shader_file_path.parent_path()};
    std::array<const char*, 1> searchPaths {module_search_path.c_str()};
    slang::SessionDesc sessionDesc {
        .targets = &targetDesc,
        .targetCount = 1,
        .searchPaths = searchPaths.data(),
        .searchPathCount = searchPaths.size()
    };

    // Create local session and add it to the global session
    // Note: local session must live for as long as we need the linked Slang shader program
    Slang::ComPtr<slang::ISession> session;
    if(SLANG_FAILED(globalSession->createSession(sessionDesc, session.writeRef()))) {
        std::println("[{}] [ERROR] Failed to create Slang local session.", mazorca::current_time());
        return std::unexpected(mazorca::error_code::invalid);
    }

    // Load Slang module from provided input file path
    // Note: we import other Slang source files automatically into a single translational unit
    Slang::ComPtr<slang::IModule> slangModule;
    {
        Slang::ComPtr<slang::IBlob> diagnosticsBlob;
        slangModule = session->loadModule(
            shader_file_path.stem().c_str(),    // Module name (internal copy makes lifetime ok?)
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

    // Identify number and names of entry points inside Slang module and store in map object
    SlangInt32 num_entry_points = slangModule->getDefinedEntryPointCount();
    std::unordered_map<std::string, Slang::ComPtr<slang::IEntryPoint>> entry_point_map;
    entry_point_map.reserve(static_cast<std::size_t>(num_entry_points));
    for (SlangInt32 i = 0; i < num_entry_points; i++) {
        Slang::ComPtr<slang::IEntryPoint> entryPoint;
        if(SLANG_FAILED(slangModule->getDefinedEntryPoint(i, entryPoint.writeRef()))) {
            std::println("[{}] [ERROR] Failed to find entry point at index: {}", mazorca::current_time(), i);
            return std::unexpected(mazorca::error_code::invalid);
        }
        // We use reflection to identify the name of the queried entry point
        std::string entry_point_name {entryPoint->getLayout()->getEntryPointByIndex(0)->getName()};
        entry_point_map.try_emplace(entry_point_name, std::move(entryPoint));
    }

    std::unordered_map<std::string, Slang::ComPtr<slang::IComponentType>> linked_program_map;
    linked_program_map.reserve(static_cast<std::size_t>(num_entry_points));
    for (const auto& [name, entry] : entry_point_map) {
        std::println("[{}] [INFO] Compiling Slang module with entry point name: {}", mazorca::current_time(), name);
    
        // Compose the Slang module
        // Note: the number of components is the number of corresponding Slang modules plus the number of corresponding entry points
        std::array<slang::IComponentType*, 2> component_types {slangModule, entry};
        Slang::ComPtr<slang::IComponentType> composed_program;
        {
            Slang::ComPtr<slang::IBlob> diagnosticsBlob;
            SlangResult result = session->createCompositeComponentType(
                component_types.data(),
                static_cast<SlangInt>(component_types.size()),
                composed_program.writeRef(),
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
        Slang::ComPtr<slang::IComponentType> linked_program;
        {
            Slang::ComPtr<slang::IBlob> diagnosticsBlob;
            SlangResult result = composed_program->link(
                linked_program.writeRef(),
                diagnosticsBlob.writeRef());
            if (diagnosticsBlob != nullptr) {
                std::println("{}", static_cast<const char*>(diagnosticsBlob->getBufferPointer()));
            }
            if (SLANG_FAILED(result)) {
                std::println("[{}] [ERROR] Failed to link the composed Slang program.", mazorca::current_time());   
                return std::unexpected(mazorca::error_code::invalid);
            }
        }
    
        linked_program_map.try_emplace(name, std::move(linked_program));
    }

    // Running the composed and linked Slang program on device
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

    for (const auto& [name, linked_program] : linked_program_map) {

        rhi::ShaderProgramDesc shader_program_desc {
            .slangGlobalScope = linked_program.get()
        };

        Slang::ComPtr<rhi::IShaderProgram> shader_program;
        {
            Slang::ComPtr<slang::IBlob> diagnosticsBlob;
            SlangResult result = device->createShaderProgram(
                shader_program_desc,
                shader_program.writeRef(),
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
            .program = shader_program.get()
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
    }

    return {};
}
