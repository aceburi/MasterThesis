#include "Analyze.hpp"

#include <vector>
#include <fstream>
#include <cstdlib>

#include "Path.hpp"

nlohmann::json analytics::analyze(const StorageAdaptor& storage) {
    // load paths
    auto paths = storage.getPaths();

    // analyze stuff
    const size_t numClasses = paths.front().leaf.possClasses().size();

    struct CountingClass {
        const size_t numClasses;
        std::vector<size_t> perLeaf;
        std::vector<size_t> perVisit;

        explicit CountingClass(size_t numClasses)
            : numClasses(numClasses), perLeaf(numClasses), perVisit(numClasses) {}
    
        void inc(const path::Path& p) {
            const auto poss = p.leaf.possClasses();
            for (size_t i = 0; i < numClasses; ++i) {
                if (poss[i] == true) {
                    perLeaf[i] += 1;
                    perVisit[i] += p.visitFreq;
                } 
            }
        }
    };

    auto convert = [](const std::vector<CountingClass>& vec) {
        std::vector<std::vector<size_t>> a, b;

        for (size_t i = 1; i < vec.size(); ++i) {  // FIXME
            auto cc = vec[i];
            a.push_back(cc.perLeaf);
            b.push_back(cc.perVisit);
        }

        return std::make_pair(a, b);
    };


    // how often has a leaf x const classes
    std::vector<CountingClass> numOfPossClasses(numClasses+1, CountingClass{numClasses});

    size_t totLeaves = 0, totVisit = 0;
    for (const auto& p : paths) {
        auto poss = p.leaf.possClasses();

        // count the number of possible classes for this leaf
        auto npc = std::count(poss.begin(), poss.end(), true);
        numOfPossClasses[npc].inc(p);

        // totals
        totLeaves += 1;
        totVisit += p.visitFreq;
    }


    // build response
    const auto [nopLeaf, nopVisit] = convert(numOfPossClasses);

    return nlohmann::json{
        {"nopLeaf", nopLeaf},
        {"nopVisit", nopVisit},
        {"totalLeaves", totLeaves},
        {"totalVisits", totVisit}
    };
}