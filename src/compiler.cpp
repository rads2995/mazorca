#include "compiler.hpp"

#include <array>
#include <iostream>

#include <slang.h>
#include <slang-com-ptr.h>
#include <slang-com-helper.h>

const char* shortestShader =
"RWStructuredBuffer<float> result;"
"[shader(\"compute\")]"
"[numthreads(1,1,1)]"
"void computeMain(uint3 threadId : SV_DispatchThreadID)"
"{"
"    result[threadId.x] = threadId.x;"
"}";

int mazorca::compiler::test() {

    // Create global session (Slang API implementation)
    Slang::ComPtr<slang::IGlobalSession> globalSession;
    createGlobalSession(globalSession.writeRef());

    // Create session
    slang::SessionDesc sessionDesc = {};

    // List of enabled compilation targets
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_SPIRV;
    targetDesc.profile = globalSession->findProfile("spirv_1_5");

    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;

    // Pre-processor defines
    std::array<slang::PreprocessorMacroDesc, 2> preprocessorMacroDesc = {
        {
            {"BIAS_VALUE", "1138"},
            {"OTHER_MACRO", "float"}
        }
    };
    sessionDesc.preprocessorMacros = preprocessorMacroDesc.data();
    sessionDesc.preprocessorMacroCount = preprocessorMacroDesc.size();

    // Compiler options
    std::array<slang::CompilerOptionEntry, 1> options = {
        {
            {
                slang::CompilerOptionName::EmitSpirvDirectly,
                {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr}
            }
        }
    };
    sessionDesc.compilerOptionEntries = options.data();
    sessionDesc.compilerOptionEntryCount = options.size();

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
            return -1;
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
            return -1;
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
        SLANG_RETURN_ON_FAIL(result);
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
        SLANG_RETURN_ON_FAIL(result);
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
        SLANG_RETURN_ON_FAIL(result);
    }

    std::cout << "Compiled " << spirvCode->getBufferSize() << " bytes of SPIR-V" << std::endl;

    return 0;
}
