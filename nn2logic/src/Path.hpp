#pragma once

#include <memory>
#include <vector>
#include <set>
#include <cassert>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <nlohmann/json.hpp>
#include <optional>
#include <bitset>
#include <list>

#ifndef MAX_PATH_LENGTH
#include <boost/dynamic_bitset.hpp>
#endif

namespace path {

struct Decision;

class InfeasibleSubsystem final {
    std::set<std::pair<size_t,size_t>> m_iisNodes = {};
public:
    InfeasibleSubsystem() {}
    explicit InfeasibleSubsystem(const std::set<std::pair<size_t,size_t>>& iisNodes) : m_iisNodes(iisNodes) {}

    void add(size_t layerIdx, size_t neuronIdx) {
        m_iisNodes.emplace(layerIdx, neuronIdx);
    }

    void add(const Decision& dec);
    void add(std::shared_ptr<Decision> ptr);

    [[nodiscard]] size_t size() const {
        return m_iisNodes.size();
    } 

    [[nodiscard]] bool empty() const {
        return m_iisNodes.empty();
    }

    [[nodiscard]] std::set<std::pair<size_t,size_t>> iisIDxs() const {
        return m_iisNodes;
    }

    [[nodiscard]] bool contains(const std::pair<size_t,size_t>& idx) const {
        return m_iisNodes.find(idx) != m_iisNodes.end();
    }

    /*
     * Iterator sugar
     */
    typedef std::set<std::pair<size_t,size_t>>::const_iterator iterator_t;

    [[nodiscard]] iterator_t begin() const {
        return m_iisNodes.cbegin();
    }

    [[nodiscard]] iterator_t end() const {
        return m_iisNodes.cend();
    }

    /*
     * Conversion for legacy code
     */
    explicit operator std::set<std::pair<size_t,size_t>>() const {
        return iisIDxs();
    }

    /*
     * Comparisons
     */
    friend bool operator==(const InfeasibleSubsystem& a, const InfeasibleSubsystem& b) {
        return a.m_iisNodes == b.m_iisNodes;
    }

    friend bool operator<(const InfeasibleSubsystem& a, const InfeasibleSubsystem& b) {
        auto aIt = a.begin();
        auto bIt = b.begin();

        while (aIt != a.end() && bIt != b.end()) {
            if (!(*aIt == *bIt)) return *aIt < *bIt;  // find the first one where they are not equal

            ++aIt;
            ++bIt;
        }

        // handle IISs of different lengths
        return a.size() < b.size();
    }
};


class MultInfeasSubsystem final {
    std::list<InfeasibleSubsystem> m_systems = {};
public:
    MultInfeasSubsystem() {}
    explicit MultInfeasSubsystem(const InfeasibleSubsystem& is) : m_systems({is}) {
        assert(!is.empty());
    }

    void add(const InfeasibleSubsystem& is) {
        assert(!is.empty());

        /*
         * wee need to ensure ordering of the systems for comparison
         * and also do deduplication
         */
        bool duplicate = false;
        // find the first one where "is" is larger than "m" -> m < is
        auto pos = std::find_if(m_systems.begin(), m_systems.end(), [&](const InfeasibleSubsystem& m) -> bool {
            if (m == is) {
                duplicate = true;
                return true;
            }

            return m < is;
        });

        if (!duplicate) {
            m_systems.insert(pos, is);
        }
    }

    void add(const MultInfeasSubsystem& other) {
        assert(!other.empty());
        for (const auto& o : other) {
            add(o);
        }
    }

    [[nodiscard]] size_t size() const {
        return m_systems.size();
    }

    [[nodiscard]] bool empty() const {
        return m_systems.empty();
    }

    [[nodiscard]] bool contains(const std::pair<size_t,size_t>& idx) const {
        return std::any_of(m_systems.begin(), m_systems.end(), [&](const auto& is) -> bool {
            return is.contains(idx);
        });
    }

    /*
     * Iterator sugar
     */
    typedef std::list<InfeasibleSubsystem>::const_iterator iterator_t;

    [[nodiscard]] iterator_t begin() const {
        return m_systems.cbegin();
    }

    [[nodiscard]] iterator_t end() const {
        return m_systems.cend();
    }

    /*
     * Conversion for legacy code
     */
    explicit operator std::set<std::pair<size_t,size_t>>() const {
        const auto N = size();
        if (N == 0) {
            return std::set<std::pair<size_t,size_t>>();
        }

        if (N > 1) {
            throw std::runtime_error("Implicitly tried to convert Multi-IIS into a singular one, not possible.");
        }

        return m_systems.begin()->iisIDxs();
    }

    /*
     * Comparisons
     */
    friend bool operator==(const MultInfeasSubsystem& a, const MultInfeasSubsystem& b) {
        return a.m_systems == b.m_systems;
    }

    friend bool operator<(const MultInfeasSubsystem& a, const MultInfeasSubsystem& b) {
        auto aIt = a.begin();
        auto bIt = b.begin();

        while (aIt != a.end() && bIt != b.end()) {
            if (!(*aIt == *bIt)) return *aIt < *bIt;  // find the first one where they are not equal

            ++aIt;
            ++bIt;
        }

        // handle different lengths
        return a.size() < b.size();
    }
};


struct Decision final {
    size_t layerIdx;
    size_t neuronIdx;
    bool decision;
    bool isConst;

    MultInfeasSubsystem iisNodes;

    std::pair<size_t,size_t> idx() const {
        return std::make_pair(layerIdx, neuronIdx);
    }

    friend bool operator==(const Decision& a, const Decision& b) {
        return std::tie(a.layerIdx, a.neuronIdx, a.decision, a.isConst, a.iisNodes) ==
            std::tie(b.layerIdx, b.neuronIdx, b.decision, b.isConst, b.iisNodes);
    }
};

struct DecisionSorter final {
    bool operator()(const Decision& lhs, const Decision& rhs) const {
        if (lhs.layerIdx != rhs.layerIdx) return lhs.layerIdx < rhs.layerIdx;
        if (lhs.neuronIdx != rhs.neuronIdx) return lhs.neuronIdx < rhs.neuronIdx;

        if (lhs.decision != rhs.decision) return lhs.decision;

        // makes no sense otherwise
        assert(lhs.isConst == rhs.isConst);

        return false;
    };
};


class Leaf {
    std::vector<bool> m_isPossible;
    std::map<size_t,MultInfeasSubsystem> iisNodes;
public:
    explicit Leaf(const std::vector<bool> &posMax) : m_isPossible(posMax) {}
    explicit Leaf(const size_t numClasses) : m_isPossible(numClasses, true) {}

    //Leaf(const std::vector<bool> &posMax, const MultInfeasSubsystem& iisNodes) : m_isPossible(posMax), iisNodes(iisNodes) {}

    Leaf() {}

    [[nodiscard]] bool isConst() const {
        return std::count(m_isPossible.begin(), m_isPossible.end(), true) == 1;
    }

    [[nodiscard]] bool partiallyConst() const {
        return std::any_of(m_isPossible.begin(), m_isPossible.end(), [](bool b) { return !b; });
    }

    [[nodiscard]] bool isPossible(size_t idx) const {
        return m_isPossible[idx];
    }

    [[nodiscard]] std::map<size_t,MultInfeasSubsystem> getIIS() const {
        return iisNodes;
    }

    // FIXME: this function needs to go
    std::set<std::pair<size_t,size_t>> iisIDxs() const {
        if (iisNodes.empty()) return std::set<std::pair<size_t,size_t>>();

        assert(iisNodes.size() == 1);
        return static_cast<std::set<std::pair<size_t,size_t>>>(iisNodes.begin()->second);
    }

    std::vector<bool> possClasses() const {
        return m_isPossible;
    }

    void setClassInfeas(size_t idx, const MultInfeasSubsystem& iisNodes) {
        m_isPossible.at(idx) = false;
        this->iisNodes[idx].add(iisNodes);
    }
};


struct Path final {
    std::vector<Decision> decisions;
    Leaf leaf;
    size_t visitFreq = 0;


    friend bool operator<(const Path& lhs, const Path& rhs) {
        if (lhs.visitFreq == rhs.visitFreq) return lhs.decisions.size() < rhs.decisions.size();
        return lhs.visitFreq < rhs.visitFreq;
    }

    friend bool operator>(const Path& lhs, const Path& rhs) {
        return !(lhs < rhs);
    }

    std::optional<Decision> getDecision(size_t layerIdx, size_t neuronIdx) const {
        auto it = std::find_if(decisions.begin(), decisions.end(), [&](const auto& d) -> bool {
            return d.layerIdx == layerIdx && d.neuronIdx == neuronIdx;
        });

        if (it == decisions.end()) return {};
        return *it;
    }

    std::vector<Decision> getDecisions(const std::set<std::pair<size_t,size_t>>& idxs) const {
        std::vector<Decision> toRet;
        for (const auto& i : idxs) {
            toRet.push_back(getDecision(i.first, i.second).value());
        }
        return toRet;
    }

    // FIXME: this needs to go
    std::vector<Decision> iisDecisions() const {
        throw std::runtime_error("function is obsolete");
        std::vector<Decision> toRet;
        /*
        const auto iis = leaf.getIIS();
        std::copy_if(decisions.begin(), decisions.end(), std::back_inserter(toRet), [&](const auto& d) -> bool {
            return iis.contains(d.idx());
        });
        */
        return toRet;
    }

    std::vector<Decision> nonConst() const {
        std::vector<Decision> toRet;

        std::copy_if(decisions.begin(), decisions.end(), std::back_inserter(toRet), [](const auto& d) {
            return !d.isConst;
        });

        return toRet;
    }

    class PathBitset final {  /// ffuuu, layer sind unterschiedlich groß und es gibt keinen single-int identifier für decisions. brauchen auch layer sizes
#ifdef MAX_PATH_LENGTH
        std::bitset<MAX_PATH_LENGTH> m_vals;
        std::bitset<MAX_PATH_LENGTH> m_mask;
#else
        boost::dynamic_bitset<> m_vals;
        boost::dynamic_bitset<> m_mask;
#endif
        const std::vector<size_t> m_layerSizes;

    public:
        explicit PathBitset(const std::vector<size_t> layerSizes) : m_layerSizes(layerSizes) {
            assert(!layerSizes.empty());
            size_t total = 0;
            for (size_t t : layerSizes) {
                total += t;
            }
#ifdef MAX_PATH_LENGTH
            if (total > MAX_PATH_LENGTH) {
                throw std::length_error("There are too many neurons. Try increasing MAX_PATH_LENGTH (in CMakeLists.txt) and recompile.");
            }
#else
            m_vals.reserve(total);
            m_vals.reserve(total);
#endif
        }

        bool overlaps(const PathBitset& other) const {
            return ((m_vals ^ other.m_vals) & m_mask & other.m_mask).none();
        }

        void set(const std::pair<size_t,size_t>& pos, bool val) {
            // determine position in bitset
            size_t bitPos = pos.second;
            for (size_t i = 0; i < pos.first; ++i) {
                bitPos += m_layerSizes.at(i);
            }

            // set val
            m_vals[bitPos] = val;
            m_mask[bitPos] = true;
        }

        explicit PathBitset(const std::vector<size_t> layerSizes, const std::vector<Decision>& decisions) : PathBitset(layerSizes) {
            for (const auto& d : decisions) {
                set(d.idx(), d.decision);
            }
        }
    };


    PathBitset asBitset(const std::vector<size_t> layerSizes) const {
        return PathBitset(layerSizes, decisions);
    }

};

// JSON converters
void to_json(nlohmann::json& j, const Path& p);  /// Path <-> JSON
void from_json(const nlohmann::json& j, Path& p);

void to_json(nlohmann::json& j, const Decision& p);  /// Decisions <-> JSON
void from_json(const nlohmann::json& j, Decision& d);

void to_json(nlohmann::json& j, const Leaf& l);  /// Leaf <-> JSON
//void from_json(const nlohmann::json& j, Leaf& l);  // Leaf is not default constructible

void to_json(nlohmann::json& j, const InfeasibleSubsystem& is);  /// InfeasibleSubsystem <-> JSON
void from_json(const nlohmann::json& j, InfeasibleSubsystem& is);

void to_json(nlohmann::json& j, const MultInfeasSubsystem& m);  /// MultInfeasSubsystem <-> JSON
void from_json(const nlohmann::json& j, MultInfeasSubsystem& m);
}


// format a decision
template <> struct fmt::formatter<path::Decision>: formatter<std::string_view> {
    auto format(const path::Decision& d, format_context& ctx) const {
        char c = ' ';
        if (d.isConst) c = 'c';

        return formatter<string_view>::format(
            fmt::format("({},{:2d},{}{})", d.layerIdx, d.neuronIdx, d.decision ? 'T' : 'F', c), ctx);
    }
};


// takes care of output formatting of a path
template <> struct fmt::formatter<path::Path>: formatter<std::string_view> {
    auto format(const path::Path& p, format_context& ctx) const {
        auto leaf = p.leaf.isConst() ? fmt::format("c {:d}", fmt::join(p.leaf.possClasses(), "")) : "";
        auto end = fmt::format("{} -> {} {}", fmt::join(p.decisions, "-"), p.visitFreq, leaf);
        return formatter<string_view>::format(end, ctx);
    }
};