#include "Path.hpp"

/*
 * InfeasibleSubsystem
 */
void path::InfeasibleSubsystem::add(const path::Decision& dec) {
    m_iisNodes.emplace(dec.idx());
}

void path::InfeasibleSubsystem::add(std::shared_ptr<Decision> ptr) {
    m_iisNodes.emplace(ptr->idx());
}


void path::to_json(nlohmann::json& j, const path::InfeasibleSubsystem& is) {
    j = is.iisIDxs();
}

void path::from_json(const nlohmann::json& j, path::InfeasibleSubsystem& is) {
    for (const std::pair<size_t,size_t>& p : j) {
        is.add(p.first, p.second);
    }
}

/*
 * MultInfeasSubsystem
 */
void path::to_json(nlohmann::json& j, const path::MultInfeasSubsystem& m) {
    j = std::vector<path::InfeasibleSubsystem>(m.begin(), m.end());
}

void path::from_json(const nlohmann::json& j, path::MultInfeasSubsystem& m) {
    for (const path::InfeasibleSubsystem& is : j.template get<std::vector<path::InfeasibleSubsystem>>()) {
        if(!is.empty()) m.add(is);
    }
}


/*
 * Path
 */
void path::to_json(nlohmann::json& j, const path::Path& p) {
    j = nlohmann::json{
        {"visitFreq", p.visitFreq},
        {"decisions", p.decisions},
        {"leaf", p.leaf}
    };
}

void path::from_json(const nlohmann::json& j, path::Path& p) {
    j.at("visitFreq").get_to(p.visitFreq);
    j.at("decisions").get_to(p.decisions);

    // Leaf
    std::vector<bool> possClasses;
    j.at("leaf").at("possClasses").get_to(possClasses);

    p.leaf = Leaf{possClasses.size()};

    // Check if leaf has iis anyways
    if (j.at("leaf").at("iis").is_null()) return;

    for (const auto& [k, v] : j.at("leaf").at("iis").template get<std::map<std::string,path::MultInfeasSubsystem>>()) {
        p.leaf.setClassInfeas(std::stoi(k), v);
    }

    assert(p.leaf.possClasses() == possClasses);
}


/*
 * Decision
 */
void path::to_json(nlohmann::json& j, const path::Decision& d) {
    j = nlohmann::json{ 
        {"layerIdx", d.layerIdx},
        {"neuronIdx", d.neuronIdx},
        {"decision", d.decision},
        {"isConst", d.isConst},
        {"iis", d.iisNodes}
    };
}

void path::from_json(const nlohmann::json& j, path::Decision& d) {
    j.at("layerIdx").get_to(d.layerIdx);
    j.at("neuronIdx").get_to(d.neuronIdx);
    j.at("decision").get_to(d.decision);
    j.at("isConst").get_to(d.isConst);
    j.at("iis").get_to(d.iisNodes);
}


/*
 * Leaf
 */
void path::to_json(nlohmann::json& j, const path::Leaf& l) {
    j = nlohmann::json{
        {"isConst", l.isConst()},
        {"possClasses", l.possClasses()},
        {"iis", {}}
    };

    for (const auto& [k, v] : l.getIIS()) {
        j["iis"][std::to_string(k)] = v;
    }
}