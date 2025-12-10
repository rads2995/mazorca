#pragma once

#include <array>
#include <expected>
#include <fstream>
#include <print>

#include <slang.h>
#include <slang-com-ptr.h>
#include <slang-com-helper.h>

namespace mazorca {

[[nodiscard]] inline 
std::expected<std::unordered_map<std::string, Slang::ComPtr<slang::IBlob>>, error_code> 
compile_shader(std::filesystem::path& shader_file_path, Slang::ComPtr<slang::IGlobalSession>& globalSession) {

    std::ifstream shader_file(shader_file_path, std::ios::binary);

    if (!shader_file) {
        std::println("ERROR: unable to read shader file: {}", shader_file_path.string());
        return std::unexpected(error_code::invalid);
    }

    // Read shader source to string for Slang shader compiler
    std::string shader_source{
        std::istreambuf_iterator<char>(shader_file), 
        std::istreambuf_iterator<char>()
    };    

    // List of enabled compilation targets
    slang::TargetDesc targetDesc = {
        .format = SLANG_SPIRV,
        .profile = globalSession->findProfile("glsl_460")
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
            return std::unexpected(error_code::invalid);
        }
    }

    // Query entry points
    // Note: entry points must be defined in the module to be compiled
    // TODO: use reflection to gather the names and number of entry points?
    Slang::ComPtr<slang::IEntryPoint> compute_entry_point;
    Slang::ComPtr<slang::IEntryPoint> vertex_entry_point;
    Slang::ComPtr<slang::IEntryPoint> fragment_entry_point;
    {
        Slang::ComPtr<slang::IBlob> diagnosticsBlob;
        
        slangModule->findEntryPointByName("compute", compute_entry_point.writeRef());
        if (!compute_entry_point) {
            std::println("Error obtaining compute entry point");
            return std::unexpected(error_code::invalid);
        }

        slangModule->findEntryPointByName("vertex", vertex_entry_point.writeRef());
        if (!vertex_entry_point) {
            std::println("Error obtaining compute entry point");
            return std::unexpected(error_code::invalid);
        }

        slangModule->findEntryPointByName("fragment", fragment_entry_point.writeRef());
        if (!fragment_entry_point) {
            std::println("Error obtaining compute entry point");
            return std::unexpected(error_code::invalid);
        }
    }

    // Compose Modules + Entry Points
    std::array<slang::IComponentType*, 4> componentTypes = {
        slangModule,
        compute_entry_point,
        vertex_entry_point,
        fragment_entry_point
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

    // Perform reflection on compiled and linked program layout
    slang::ProgramLayout* programLayout = linkedProgram->getLayout();

    // Get target SPIR-V code and store in map hashed by entry point name
    std::unordered_map<std::string, Slang::ComPtr<slang::IBlob>> spirv_map;
    for (std::size_t i = 0; i < programLayout->getEntryPointCount(); i++) {

        Slang::ComPtr<slang::IBlob> diagnosticsBlob;
        Slang::ComPtr<slang::IBlob> spirvBlob;
        SlangResult result = linkedProgram->getEntryPointCode(
            static_cast<SlangInt>(i),   // Entry point index
            0,                          // Target index
            spirvBlob.writeRef(),
            diagnosticsBlob.writeRef());
        if (diagnosticsBlob != nullptr) {
            std::println("{}", static_cast<const char*>(diagnosticsBlob->getBufferPointer()));
        }
        if (SLANG_FAILED(result)) {
            return std::unexpected(error_code::invalid);
        }

        spirv_map[programLayout->getEntryPointByIndex(i)->getName()] = spirvBlob;
    }

    for (const auto& [key, val]: spirv_map) {
        std::println("{}, {}", key, val->getBufferSize());
    }

    return spirv_map;
}

} // namespace mazorca
