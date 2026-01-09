#include "training.hpp"
#include "compiler.hpp"

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

    return {};
}
