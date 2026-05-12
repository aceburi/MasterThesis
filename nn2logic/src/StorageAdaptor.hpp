#pragma once

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include <fmt/os.h>
#include <Eigen/Dense>

#include "Path.hpp"
#include "quant/Exchange.hpp"


class StorageAdaptor final {
public:
    typedef std::pair<std::vector<int>,int> Samples_t;

private:
    enum class JType : bool {
        REGULAR = true, BINARY = false
    };

    static JType checkExt(const std::filesystem::path& filepath) {
        if (filepath.extension() == ".json") return JType::REGULAR;
        if (filepath.extension() == ".ubj") return JType::BINARY;

        throw std::runtime_error("Invalid file extension provided");
    }

    static nlohmann::json parseByExt(const std::filesystem::path& filepath) {
        nlohmann::json j;

        // parse file
        if (checkExt(filepath) == JType::REGULAR) {
            std::ifstream ifs(filepath);
            j = nlohmann::json::parse(ifs);
            ifs.close();
        } else {
            std::ifstream ifs(filepath, std::ios::binary);
            j = nlohmann::json::from_ubjson(ifs);
            ifs.close();
        }

        return j;
    }

    static void saveByExt(const std::filesystem::path& dst, const nlohmann::json& j) {
        if (checkExt(dst) == JType::REGULAR) {
            std::ofstream o(dst);
            o << j << std::endl;
            o.close();
        } else {
            std::ofstream o(dst, std::ios::binary);
            auto ubjson = nlohmann::json::to_ubjson(j);
            o.write(reinterpret_cast<const char*>(ubjson.data()), ubjson.size());
            o.close();
        }
    }

    std::vector<path::Path> m_paths;
    std::vector<QTree::Layer<int>> m_layers;
    std::vector<Samples_t> m_samples;

public:
    explicit StorageAdaptor(const nlohmann::json& j) {
        // store
        j.at("paths").get_to(m_paths);
        j.at("layers").get_to(m_layers);

        // Old sample format
        if (j.contains("data") && j.contains("targets")) {
            std::vector<std::vector<int>> data;
            std::vector<int> targets;

            j.at("data").get_to(data);
            j.at("targets").get_to(targets);
            
            assert(data.size() == targets.size());

            for (size_t i = 0; i < data.size(); ++i) {
                m_samples.emplace_back(data[i], targets[i]);
            }

        // new sample format
        } else if (j.contains("samples")) {
            j.at("samples").get_to(m_samples);
        }
    }

    explicit StorageAdaptor(const std::string& filepath) : StorageAdaptor(parseByExt(filepath)) {}

    StorageAdaptor(const std::vector<path::Path>& paths, const std::vector<QTree::Layer<int>>& layers, const std::vector<Samples_t>& samples = {})
        : m_paths(paths), m_layers(layers), m_samples(samples) {}

    StorageAdaptor(const std::vector<QTree::Layer<int>>& layers, const std::vector<Samples_t>& samples) : m_layers(layers), m_samples(samples) {}

    // Get
    [[nodiscard]] std::vector<path::Path> getPaths() const {
        return m_paths;
    }

    [[nodiscard]] std::vector<QTree::Layer<int>> getLayers() const {
        return m_layers;
    }

    [[nodiscard]] std::vector<Samples_t> getSamples() const {
        return m_samples;
    }

    // Set
    void setLayers(const std::vector<QTree::Layer<int>>& layers) {
        m_layers = layers;
    }

    void setPaths(const std::vector<path::Path>& paths) {
        m_paths = paths;
    }

    void setDataset(const std::vector<Samples_t>& samples) {
        m_samples = samples;
    }

    // convert between json and ubjson
    static void convert(const std::string& src, const std::string& dst) {
        auto j = parseByExt(src);
        saveByExt(dst, j);
    }

    void save(const std::string& dst) const {
        nlohmann::json j;
        j["layers"] = m_layers;
        j["paths"] = m_paths;
        j["samples"] = m_samples;

        saveByExt(dst, j);
    }

    // create header
    static void createHeaderFromDataset(const std::string& dst, const std::vector<Samples_t>& samples) {        
        // targets
        std::vector<int> targets;
        targets.reserve(samples.size());
        
        // intermediate strings
        std::vector<std::string> im;
        im.reserve(samples.size());
        for (const auto& [s, t] : samples) {
            im.push_back(fmt::format("{{ {} }}", fmt::join(s, ", ")));
            targets.push_back(t);
        }

        // create file
        auto out = fmt::output_file(dst);

        // write actual file
        out.print(
            "#ifndef H_DATASET\n"
            "#define H_DATASET\n"

            "#include <stdint.h>\n"
            "#include <stdbool.h>\n"

            "const size_t datasetSize = {0};\n"

            "const uint8_t dataset[{0}][{1}] = {{\n"
            "\t{2}\n"
            "}};"

            "const uint8_t target[] = {{ {3} }};\n"

            "#endif",
            samples.size(), samples.front().first.size(),
            fmt::join(im, ",\n\t"), fmt::join(targets, ", ")
        );
    }

    void writeHeader(const std::string& dst) const {
        createHeaderFromDataset(dst, m_samples);
    }

    // Iterators
    std::vector<Samples_t>::const_iterator sbegin() const {
        return m_samples.begin();
    }

    std::vector<Samples_t>::const_iterator send() const {
        return m_samples.end();
    }
};