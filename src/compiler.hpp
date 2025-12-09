#pragma once

#include <array>
#include <expected>
#include <iostream>
#include <fstream>
#include <print>

#include <slang.h>
#include <slang-com-ptr.h>
#include <slang-com-helper.h>

namespace mazorca {

[[nodiscard]] inline std::expected<Slang::ComPtr<slang::IBlob>, error_code> compile_shader(std::filesystem::path& shader_file_path) {

    std::ifstream shader_file(shader_file_path, std::ios::binary);

    if (!shader_file) {
        return std::unexpected(error_code::invalid);
    }

    // Read shader source to string for Slang shader compiler
    std::string shader_source{
        std::istreambuf_iterator<char>(shader_file), 
        std::istreambuf_iterator<char>()
    };    

    // Create Slang global session
    // TODO: "applications are advised to use a single global session if possible,
    // rather than creating and then disposing of one for each compile." - Slang team
    Slang::ComPtr<slang::IGlobalSession> globalSession;
    slang::createGlobalSession(globalSession.writeRef());

    // List of enabled compilation targets
    slang::TargetDesc targetDesc = {
        .format = SLANG_SPIRV,
        .profile = globalSession->findProfile("glsl_460")
    };
    
    // Create session
    slang::SessionDesc sessionDesc = {
        .targets = &targetDesc,
        .targetCount = 1
    };

    // Create the session
    Slang::ComPtr<slang::ISession> session;
    globalSession->createSession(sessionDesc, session.writeRef());

    // Load modules
    Slang::ComPtr<slang::IModule> slangModule;
    {
        Slang::ComPtr<slang::IBlob> diagnosticsBlob;
        
        slangModule = session->loadModuleFromSourceString(
            shader_file_path.stem().c_str(),        // Module name
            shader_file_path.c_str(),               // Module path
            shader_source.c_str(),                  // Shader source code
            diagnosticsBlob.writeRef()              // Optional diagnostic container
        );
        if (diagnosticsBlob != nullptr) {
            std::println("{}", static_cast<const char*>(diagnosticsBlob->getBufferPointer()));
        }
        if (!slangModule) {
            return std::unexpected(error_code::invalid);
        }
    }

    // Query entry points
    Slang::ComPtr<slang::IEntryPoint> entryPoint;
    {
        Slang::ComPtr<slang::IBlob> diagnosticsBlob;
        
        slangModule->findEntryPointByName("computeMain", entryPoint.writeRef());
        if (!entryPoint) {
            std::println("Error obtaining entry point");
            return std::unexpected(error_code::invalid);
        }
    }

    // Compose Modules + Entry Points
    std::array<slang::IComponentType*, 2> componentTypes = {
        slangModule,
        entryPoint
    };

    Slang::ComPtr<slang::IComponentType> composedProgram;
    {
        Slang::ComPtr<slang::IBlob> diagnosticsBlob;
        SlangResult result = session->createCompositeComponentType(
            componentTypes.data(),
            componentTypes.size(),
            composedProgram.writeRef(),
            diagnosticsBlob.writeRef());
        if (diagnosticsBlob != nullptr) {
            std::println("{}", static_cast<const char*>(diagnosticsBlob->getBufferPointer()));
        }
        if (SLANG_FAILED(result))
            return std::unexpected(error_code::invalid);
    }

    // TODO: should we perform reflection on shader parameters and layout?

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
            return std::unexpected(error_code::invalid);
    }

    // Get Target Kernel Code
    Slang::ComPtr<slang::IBlob> spirvCode;
    {
        Slang::ComPtr<slang::IBlob> diagnosticsBlob;
        SlangResult result = linkedProgram->getEntryPointCode(
            0,  // 0 means only one entry point
            0,  // 0 means only one target
            spirvCode.writeRef(),
            diagnosticsBlob.writeRef());
        if (diagnosticsBlob != nullptr) {
            std::println("{}", static_cast<const char*>(diagnosticsBlob->getBufferPointer()));
        }
        if (SLANG_FAILED(result))
            return std::unexpected(error_code::invalid);
    }

    std::println(
        "Compiled {} bytes of SPIR-V from shader source string", 
        spirvCode->getBufferSize()
    );

    return spirvCode;
}

} // namespace mazorca
