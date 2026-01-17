#pragma once

#include "mazorca/mazorca.hpp"
#include "graphics.hpp"

#include <slang.h>
#include <slang-com-ptr.h>

namespace mazorca {

[[nodiscard]] 
std::expected<void, mazorca::error_code>
compile_shader(mazorca::vulkan_data vulkan_data, const std::filesystem::path& shader_file_path, const Slang::ComPtr<slang::IGlobalSession>& globalSession);

} // namespace mazorca
