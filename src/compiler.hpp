#pragma once

#include <slang-com-ptr.h>
#include <slang.h>

#include "graphics.hpp"
#include "mazorca/mazorca.hpp"

namespace mazorca {

[[nodiscard]]
constexpr auto compile_shader(const std::filesystem::path& shader_file_path,
                              const Slang::ComPtr<slang::IGlobalSession>& globalSession)
    -> std::expected<std::unordered_map<std::string, Slang::ComPtr<slang::IBlob>>, mazorca::error_code> {
    // List of enabled compilation targets
    slang::TargetDesc const targetDesc{.format = SLANG_SPIRV, .profile = globalSession->findProfile("spirv_1_6")};

    // Create session descriptor
    // Note: create new local path object of file parent path due to lifetime of c_str
    std::filesystem::path const module_search_path{shader_file_path.parent_path()};
    std::array<const char*, 1> searchPaths{module_search_path.c_str()};
    slang::SessionDesc const sessionDesc{.targets = &targetDesc,
                                         .targetCount = 1,
                                         .searchPaths = searchPaths.data(),
                                         .searchPathCount = searchPaths.size()};

    // Create local session and add it to the global session
    // Note: local session must live for as long as we need the linked Slang shader program
    Slang::ComPtr<slang::ISession> session;
    if (SLANG_FAILED(globalSession->createSession(sessionDesc, session.writeRef()))) {
        std::println("[{}] [ERROR] Failed to create Slang local session.", mazorca::current_time());
        return std::unexpected(mazorca::error_code::invalid);
    }

    // Load Slang module from provided input file path
    // Note: we import other Slang source files automatically into a single translational unit
    Slang::ComPtr<slang::IModule> slangModule;
    {
        Slang::ComPtr<slang::IBlob> diagnosticsBlob;
        slangModule =
            session->loadModule(shader_file_path.stem().c_str(),  // Module name (internal copy makes lifetime ok?)
                                diagnosticsBlob.writeRef()        // Optional diagnostic container
            );
        if (diagnosticsBlob != nullptr) {
            std::println("{}", static_cast<const char*>(diagnosticsBlob->getBufferPointer()));
        }
        if (slangModule == nullptr) {
            std::println("[{}] [ERROR] Failed to load Slang module into local session.", current_time());
            return std::unexpected(mazorca::error_code::invalid);
        }
    }

    // Identify number and names of entry points inside Slang module and store in map object
    SlangInt32 const num_entry_points = slangModule->getDefinedEntryPointCount();
    std::unordered_map<std::string, Slang::ComPtr<slang::IEntryPoint>> entry_point_map;
    entry_point_map.reserve(static_cast<std::size_t>(num_entry_points));
    for (SlangInt32 i = 0; i < num_entry_points; i++) {
        Slang::ComPtr<slang::IEntryPoint> entryPoint;
        if (SLANG_FAILED(slangModule->getDefinedEntryPoint(i, entryPoint.writeRef()))) {
            std::println("[{}] [ERROR] Failed to find entry point at index: {}", mazorca::current_time(), i);
            return std::unexpected(mazorca::error_code::invalid);
        }
        // We use reflection to identify the name of the queried entry point
        std::string const entry_point_name{entryPoint->getLayout()->getEntryPointByIndex(0)->getName()};
        entry_point_map.try_emplace(entry_point_name, std::move(entryPoint));
    }

    std::unordered_map<std::string, Slang::ComPtr<slang::IComponentType>> linked_program_map;
    linked_program_map.reserve(static_cast<std::size_t>(num_entry_points));
    for (const auto& [name, entry] : entry_point_map) {
        std::println("[{}] [INFO] Compiling Slang module for entry point name: {}", mazorca::current_time(), name);

        // For each entry point, compose and link the Slang program
        Slang::ComPtr<slang::IComponentType> linked_program;
        {
            Slang::ComPtr<slang::IBlob> diagnosticsBlob;
            SlangResult const result = entry->link(linked_program.writeRef(), diagnosticsBlob.writeRef());
            if (diagnosticsBlob != nullptr) {
                std::println("{}", static_cast<const char*>(diagnosticsBlob->getBufferPointer()));
            }
            if (SLANG_FAILED(result)) {
                std::println(
                    "[{}] [ERROR] Failed to link Slang program with entry point name: "
                    "{}",
                    mazorca::current_time(), name);
                return std::unexpected(mazorca::error_code::invalid);
            }
        }
        linked_program_map.try_emplace(name, std::move(linked_program));
    }

    // Extract the SPIR-V binary for each of the entry points
    std::unordered_map<std::string, Slang::ComPtr<slang::IBlob>> spirv_map;
    for (const auto& [name, program] : linked_program_map) {
        std::println("[{}] [INFO] Extracting SPIR-V blob for entry point name: {}", mazorca::current_time(), name);

        // For each entry point, obtain the entry point SPIR-V binary code
        Slang::ComPtr<slang::IBlob> spirv_blob;
        {
            Slang::ComPtr<slang::IBlob> diagnosticsBlob;
            SlangResult const result =
                program->getEntryPointCode(0,  // Each linked program contains at most one entry point
                                           0,  // Each linked program contains at most one target
                                           spirv_blob.writeRef(), diagnosticsBlob.writeRef());
            if (diagnosticsBlob != nullptr) {
                std::println("{}", static_cast<const char*>(diagnosticsBlob->getBufferPointer()));
            }
            if (SLANG_FAILED(result)) {
                std::println(
                    "[{}] [ERROR] Failed to link Slang program with entry point name: "
                    "{}",
                    mazorca::current_time(), name);
                return std::unexpected(mazorca::error_code::invalid);
            }
        }
        std::println("[{}] [INFO] Extracted {} bytes for program with entry point name: {}", mazorca::current_time(),
                     spirv_blob->getBufferSize(), name);
        spirv_map.try_emplace(name, std::move(spirv_blob));
    }
    return spirv_map;
}

}  // namespace mazorca
