#include <array>
#include <expected>
#include <iostream>

#include <slang.h>
#include <slang-com-ptr.h>
#include <slang-com-helper.h>

namespace mazorca {

const char* shortestShader =
"RWStructuredBuffer<float> result;"
"[shader(\"compute\")]"
"[numthreads(1,1,1)]"
"void computeMain(uint3 threadId : SV_DispatchThreadID)"
"{"
"    result[threadId.x] = threadId.x;"
"}";

[[nodiscard]] inline std::expected<Slang::ComPtr<slang::IBlob>, error_code> compile_shader() {

    // Create global session
    Slang::ComPtr<slang::IGlobalSession> globalSession;
    SlangGlobalSessionDesc desc{
        .enableGLSL = true
    };
    slang::createGlobalSession(&desc, globalSession.writeRef());

    // List of enabled compilation targets
    slang::TargetDesc targetDesc = {
        .format = SLANG_SPIRV,
        .profile = globalSession->findProfile("glsl_460")
    };

    // Compiler options
    std::array<slang::CompilerOptionEntry, 1> options = {
        {
            {
                slang::CompilerOptionName::EmitSpirvDirectly,
                {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr}
            }
        }
    };
    
    // Create session
    slang::SessionDesc sessionDesc = {
        .targets = &targetDesc,
        .targetCount = 1,
        .compilerOptionEntries = options.data(),
        .compilerOptionEntryCount = options.size()
    };

    // Create the session
    Slang::ComPtr<slang::ISession> session;
    globalSession->createSession(sessionDesc, session.writeRef());

    // Load modules
    Slang::ComPtr<slang::IModule> slangModule;
    {
        Slang::ComPtr<slang::IBlob> diagnosticsBlob;
        slangModule = session->loadModuleFromSourceString(
            "shortest",                 // Module name
            "shortest.slang",           // Module path
            shortestShader,             // Shader source code
            diagnosticsBlob.writeRef()  // Optional diagnostic container
        );
        if (diagnosticsBlob != nullptr) {
            std::cout 
                << static_cast<const char*>(diagnosticsBlob->getBufferPointer()) 
                << '\n';
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
        if (!entryPoint)
        {
            std::cout << "Error getting entry point" << std::endl;
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
            std::cout 
                << static_cast<const char*>(diagnosticsBlob->getBufferPointer()) 
                << '\n';
        }
        if (SLANG_FAILED(result))
            return std::unexpected(error_code::invalid);;
    }

    // Link
    Slang::ComPtr<slang::IComponentType> linkedProgram;
    {
        Slang::ComPtr<slang::IBlob> diagnosticsBlob;
        SlangResult result = composedProgram->link(
            linkedProgram.writeRef(),
            diagnosticsBlob.writeRef());
        if (diagnosticsBlob != nullptr) {
            std::cout 
                << static_cast<const char*>(diagnosticsBlob->getBufferPointer()) 
                << '\n';
        }
        if (SLANG_FAILED(result))
            return std::unexpected(error_code::invalid);
    }

    // Get Target Kernel Code
    Slang::ComPtr<slang::IBlob> spirvCode;
    {
        Slang::ComPtr<slang::IBlob> diagnosticsBlob;
        SlangResult result = linkedProgram->getEntryPointCode(
            0,
            0,
            spirvCode.writeRef(),
            diagnosticsBlob.writeRef());
        if (diagnosticsBlob != nullptr) {
            std::cout 
                << static_cast<const char*>(diagnosticsBlob->getBufferPointer()) 
                << '\n';
        }
        if (SLANG_FAILED(result))
            return std::unexpected(error_code::invalid);
    }

    std::cout << "Compiled " << spirvCode->getBufferSize() << " bytes of SPIR-V" << std::endl;

    return spirvCode;
}

} // namespace mazorca
