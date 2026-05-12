#pragma once

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <thread>

class Settings final {
public:
    /// threads that should be used by gurobi
    static const size_t gurobiThreads = 2;

    /// maximum number of threads that should be used
    static size_t maxThreads() {
        return std::max<size_t>(std::thread::hardware_concurrency()-1, 8);
    }

    /// Concurrent worker threads that each hold a GRBEnv / GRBModel copy (SimplePool stages).
    /// Academic / single-use Gurobi licenses typically allow very few simultaneous environments;
    /// defaulting high concurrency here has produced native abort() mid-run on Windows.
    /// Override at runtime: NN2LOGIC_MAX_GUROBI_POOL_THREADS (minimum 1).
    static size_t maxSpGurobi() {
        if (const char* raw = std::getenv("NN2LOGIC_MAX_GUROBI_POOL_THREADS")) {
            char* end = nullptr;
            unsigned long v = std::strtoul(raw, &end, 10);
            if (end != raw && v >= 1) {
                return static_cast<size_t>(v);
            }
        }
        return 1;
    }
};