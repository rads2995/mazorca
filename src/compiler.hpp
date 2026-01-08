#pragma once

#include "mazorca/mazorca.hpp"

#include <unordered_map>
#include <string>

#include <slang.h>
#include <slang-com-ptr.h>

namespace mazorca {

[[nodiscard]] 
std::expected<std::unordered_map<std::string, Slang::ComPtr<slang::IBlob>>, error_code> 
compile_shader(const std::filesystem::path& shader_file_path, Slang::ComPtr<slang::IGlobalSession>& globalSession);

} // namespace mazorca
