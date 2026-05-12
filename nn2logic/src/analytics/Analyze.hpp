#pragma once

#include <nlohmann/json.hpp>
#include <string>

#include "StorageAdaptor.hpp"

namespace analytics {

nlohmann::json analyze(const StorageAdaptor& storage);

}