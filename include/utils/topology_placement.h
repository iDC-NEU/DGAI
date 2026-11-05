#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace pipeann {
// neighbors is the distance-sorted, pruned outgoing list. Only the first three
// neighbors nominate pages; every outgoing neighbor contributes to their scores.
template<class Neighbors, class PageOf>
std::vector<std::pair<uint64_t, double>> topology_placement_pages(
    const Neighbors &neighbors, PageOf page_of, uint64_t invalid_page) {
  std::vector<std::pair<uint64_t, double>> pages;
  pages.reserve(3);
  for (size_t i = 0; i < std::min<size_t>(3, neighbors.size()); ++i) {
    const uint64_t page = page_of(neighbors[i].id);
    if (page != invalid_page && std::find_if(pages.begin(), pages.end(),
        [page](const auto &p) { return p.first == page; }) == pages.end())
      pages.emplace_back(page, 0.0);
  }
  if (!neighbors.empty()) {
    const double lo = neighbors.front().distance, hi = neighbors.back().distance;
    for (const auto &neighbor : neighbors) {
      const uint64_t page = page_of(neighbor.id);
      for (auto &candidate : pages)
        if (candidate.first == page)
          candidate.second += hi > lo ? 1.0 - (neighbor.distance - lo) / (hi - lo) : 1.0;
    }
  }
  // Stable ties retain the nearest nominating neighbor's page.
  std::stable_sort(pages.begin(), pages.end(),
                   [](const auto &a, const auto &b) { return a.second > b.second; });
  return pages;
}
}  // namespace pipeann
