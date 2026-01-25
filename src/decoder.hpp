#pragma once

#include "mazorca/mazorca.hpp"
#include "vgf/decoder.h"

namespace mazorca {

[[nodiscard]] 
std::expected<void, mazorca::error_code> decode_graph(const std::filesystem::path& vgf_file_path);

} // namespace mazorca
