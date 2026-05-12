#pragma once

#include <filesystem>
#include <string>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <fmt/format.h>
#include <iostream>
#include <vector>
#include <thread>
#include <fmt/ranges.h>

#include "StorageAdaptor.hpp"


namespace QSim {


struct ExecResult final {
    const int status;
    const std::string output;
};


ExecResult executeCommand(const std::string& cmd);


class CompilerAdaptor final {
    std::filesystem::path m_dir;

public:
    CompilerAdaptor() {
        auto tmp_dir = std::filesystem::temp_directory_path();
        m_dir = tmp_dir / "LeTree_Sim" / std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()));

        if (std::filesystem::exists(m_dir)) {  // check if directory already exists
            std::filesystem::remove_all(m_dir);
        }

        std::filesystem::create_directories(m_dir);  // create directory

        // initialize with templates
        // TODO check if local exists and fall back to BUILD_DIR version
        std::filesystem::copy(std::filesystem::path(BUILD_DIR) / "template", m_dir, std::filesystem::copy_options::recursive);
    }

    // pass training set
    explicit CompilerAdaptor(const StorageAdaptor& storage) : CompilerAdaptor() {
        auto filePath = m_dir / "dataset.h";
        storage.writeHeader(filePath.string());
    }

    ~CompilerAdaptor() {
        std::filesystem::remove_all(m_dir);
    }

    void addFile(const std::filesystem::path& src, const std::string& dst = "") const;

    bool compile() const {
        //auto res = executeCommand(fmt::format("/bin/bash -c 'cd \"{}\";PATH=$PATH:/opt/riscv/bin make &> compilelog.txt'", m_dir.string()));
        auto res = executeCommand(fmt::format("/bin/bash -c 'cd \"{}\";PATH=$PATH:/opt/riscv/bin make &> /dev/null'", m_dir.string()));
        
        if (res.status < 0)
            throw std::runtime_error(res.output);


        return res.status >= 0;
    }


    std::filesystem::path get() const {
        auto f = m_dir / "test.elf";
        if (!std::filesystem::exists(f)) {
            fmt::print("tried using {} but did not exist\n", f.string());
            assert(false);
        }
        return f;
    }

    bool removeFile(const std::string& name) const {
        return std::filesystem::remove(m_dir / name);
    }
};
}