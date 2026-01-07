#pragma once

#include <slang.h>
#include <slang-com-ptr.h>
#include <slang-rhi.h>

namespace mazorca {

[[nodiscard]] inline 
std::expected<std::unordered_map<std::string, Slang::ComPtr<slang::IBlob>>, error_code> 
train_model(std::filesystem::path& shader_file_path, Slang::ComPtr<slang::IGlobalSession>& globalSession) {

}

}
