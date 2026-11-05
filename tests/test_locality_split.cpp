#include "utils/locality_split.h"
#include "utils/prune_neighbors.h"
#include "utils/topology_placement.h"
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

void check(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

int main() {
  try {
    // Export must preserve the graph's ID list and use the original distances,
    // including saturation entries that are nearer than some selected entries.
    for (bool saturate : {false, true}) {
      pipeann::IndexBuildParameters params;
      params.R = 4; params.C = 20; params.alpha = 1.2f;
      params.saturate_graph = saturate;
      std::vector<pipeann::Neighbor> pool;
      for (unsigned id = 0; id < 10; ++id) pool.emplace_back(id, float(id + 1), true);
      auto copy = pool;
      std::vector<uint32_t> old_ids, ids;
      std::vector<pipeann::Neighbor> selected;
      size_t old_calls = 0, calls = 0;
      pipeann::prune_neighbors(copy, old_ids, params, pipeann::Metric::L2,
          [&](uint32_t, uint32_t) { ++old_calls; return 0.01f; });
      pipeann::prune_neighbors(pool, ids, params, pipeann::Metric::L2,
          [&](uint32_t, uint32_t) { ++calls; return 0.01f; }, &selected);
      check(old_ids == ids && calls == old_calls, "export changed pruning or distance work");
      check(selected.size() == ids.size() && std::is_sorted(selected.begin(), selected.end()), "bad export order");
      for (const auto &n : selected) {
        check(std::find(ids.begin(), ids.end(), n.id) != ids.end(), "exported unselected ID");
        check(n.distance == float(n.id + 1), "exported distance changed");
      }
    }
    // Top-3 means three neighbors, not three distinct pages; all selected
    // neighbors contribute and score ties preserve nomination order.
    std::vector<pipeann::Neighbor> neighbors;
    for (unsigned id = 0; id < 5; ++id) neighbors.emplace_back(id, float(id), true);
    const uint64_t mapping[] = {10, 20, 20, 30, 20};
    auto pages = pipeann::topology_placement_pages(neighbors,
        [&](uint32_t id) { return mapping[id]; }, UINT64_MAX);
    check(pages.size() == 2 && pages[0].first == 20 && pages[0].second == 1.25, "wrong page scoring");
    neighbors.resize(3);
    for (auto &n : neighbors) n.distance = 1;
    pages = pipeann::topology_placement_pages(neighbors,
        [](uint32_t id) { return uint64_t(id + 10); }, UINT64_MAX);
    check(pages[0].first == 10 && pages[1].first == 11, "unstable equal-distance ties");
    std::mt19937 random(3378);
    for (size_t n = 2; n <= 32; ++n) {
      std::vector<uint32_t> ids(n);
      std::iota(ids.begin(), ids.end(), 0);
      std::vector<std::vector<uint32_t>> adjacency(n);
      for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
          if (i != j && random() % 3 == 0) adjacency[i].push_back(j);
      auto distance = [](uint32_t a, uint32_t b) { return double(a > b ? a - b : b - a); };
      auto result = pipeann::locality_split::weighted_bisection(ids, adjacency, n - 1, distance);
      check(result.group == pipeann::locality_split::weighted_bisection(ids, adjacency, n - 1, distance).group,
            "METIS repeat nondeterminism");
      size_t a = std::count(result.group.begin(), result.group.end(), 0);
      check(result.group.size() == n && a > 0 && a < n, "empty METIS group");
      // Duplicate/self/external adjacency does not add graph connectivity.
      auto dirty = adjacency;
      dirty[0].push_back(0); dirty[0].push_back(999);
      if (!adjacency[0].empty()) dirty[0].push_back(adjacency[0].front());
      check(result.group == pipeann::locality_split::weighted_bisection(ids, dirty, n - 1, distance).group,
            "duplicate/self/external edge changed partition");
    }
    // Known disconnected clusters have a zero-cut balanced partition.
    std::vector<uint32_t> ids{0,1,2,3,4,5};
    std::vector<std::vector<uint32_t>> adjacency{{1,2},{0,2},{0,1},{4,5},{3,5},{3,4}};
    auto r = pipeann::locality_split::weighted_bisection(ids, adjacency, 5,
        [](uint32_t, uint32_t) { return 1.0; });
    check(r.group[0] == r.group[1] && r.group[1] == r.group[2] &&
          r.group[3] == r.group[4] && r.group[4] == r.group[5] && r.group[0] != r.group[3], "lost zero-cut clusters");
    std::cout << "ALL_LAYOUT_REUSE_TESTS_PASS\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
