#pragma once

#include "mazorca/mazorca.hpp"

#include <slang.h>
#include <slang-com-ptr.h>

namespace mazorca {

[[nodiscard]] 
std::expected<void, mazorca::error_code>
compile_shader(const std::filesystem::path& shader_file_path, Slang::ComPtr<slang::IGlobalSession>& globalSession);

} // namespace mazorca
