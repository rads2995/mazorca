#pragma once

#include "mazorca/mazorca.hpp"
#include "vgf/decoder.h"

namespace mazorca {

[[nodiscard]] 
inline constexpr std::expected<void, mazorca::error_code> decode_graph(const std::filesystem::path& vgf_file_path) {
    
    std::ifstream vgf_file(vgf_file_path, std::ios::binary);

    if (!vgf_file) {
        std::println("[{}] [ERROR] Unable to read VGF input file: {}", mazorca::current_time(), vgf_file_path.string());
        return std::unexpected(mazorca::error_code::invalid);
    }

    return {};
}

} // namespace mazorca
