#include <sstream>
#include <stdio.h>
#include "CompilerAdaptor.hpp"


QSim::ExecResult QSim::executeCommand(const std::string& cmd) {
    char buffer[128];
    std::stringstream result;
#ifdef _WIN32
    FILE* pipe = _popen(cmd.c_str(), "r");
#else
    FILE* pipe = popen(cmd.c_str(), "r");
#endif

    if (!pipe) throw std::runtime_error("popen() failed!");
    
    try {
        while (fgets(buffer, 128, pipe) != NULL) {
            result << buffer;
        }
    } catch (...) {
#ifdef _WIN32
        _pclose(pipe);
#else
        pclose(pipe);
#endif
        throw;
    }
#ifdef _WIN32
    int status = _pclose(pipe);
#else
    int status = pclose(pipe);
#endif
    return ExecResult{
        .status = status,
        .output = result.str()
    };
}


void QSim::CompilerAdaptor::addFile(const std::filesystem::path& src, const std::string& dst) const {
    auto copyTo = m_dir;

    if (dst.empty()) {
        copyTo /= src.filename();
    } else {
        std::stringstream ss;
        ss << dst;

        std::string buf;
        const size_t depth = std::count(dst.begin(), dst.end(), '/');
        for (size_t i = 0; std::getline(ss, buf, '/'); ++i) {
            copyTo /= buf;

            if (i > depth) std::filesystem::create_directory(copyTo);
        }
    }

    assert(std::filesystem::copy_file(src, copyTo, std::filesystem::copy_options::overwrite_existing));

    // take care of object file
    //std::filesystem::path o(".o");
    //copyTo.replace_extension(o);
    //std::filesystem::remove(copyTo);
}