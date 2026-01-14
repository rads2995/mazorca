#pragma once

#include "mazorca/mazorca.hpp"

#include <slang.h>
#include <slang-com-ptr.h>

namespace mazorca {

[[nodiscard]] 
std::expected<void, mazorca::error_code> 
train_model(Slang::ComPtr<slang::IComponentType>& slang_program, Slang::ComPtr<slang::IGlobalSession>& globalSession);

}
