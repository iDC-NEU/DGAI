#pragma once

#include <algorithm>
#include <metis.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pipeann {
namespace locality_split {

// Distance-weighted minimum cut via METIS, with a 1.03 balance tolerance.
struct Diagnostics {
  uint64_t distance_evals = 0;
  uint64_t directed_edges = 0;
  uint64_t undirected_edges = 0;
  uint64_t zero_degree_nodes = 0;
  uint64_t initializations = 0;
  uint64_t sweeps = 0;
  uint64_t accepted_moves = 0;
  uint64_t cache_trials = 0;
  uint64_t cache_rejections = 0;
  uint64_t row_recomputations = 0;
  uint64_t boundary_recomputations = 0;
  bool no_edge_fallback = false;
  double normalized_cut = 0;
  double distance_cost = 0;
};

struct Result {
  std::vector<uint8_t> group;
  Diagnostics diagnostics;
};

// ids.back() is the newly inserted node; adjacency contains snapshots of the
// actual graph lists at the layout-assignment point. Only edges within ids
// contribute. distance(a,b) is the existing in-memory vector distance routine.
template<class Distance>
Result weighted_bisection(const std::vector<uint32_t> &ids,
                          const std::vector<std::vector<uint32_t>> &adjacency,
                          size_t capacity, Distance distance, unsigned max_iterations = 10) {
  const size_t n = ids.size();
  if (n < 2 || n > 2 * capacity || adjacency.size() != n)
    throw std::invalid_argument("locality split: invalid capacity or input size");
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < i; ++j)
      if (ids[i] == ids[j]) throw std::invalid_argument("locality split: duplicate node");

  Result result;
  std::vector<double> distances(n * n, 0.0);
  std::vector<uint8_t> directed(n * n, 0);
  for (size_t i = 0; i < n; ++i) {
    for (uint32_t neighbor : adjacency[i]) {
      const auto pos = std::find(ids.begin(), ids.end(), neighbor);
      if (pos == ids.end()) continue;
      const size_t j = static_cast<size_t>(pos - ids.begin());
      if (i != j) directed[i * n + j] = 1;
    }
  }
  double edge_min = std::numeric_limits<double>::infinity();
  double edge_max = -std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      const unsigned multiplicity = directed[i * n + j] + directed[j * n + i];
      if (!multiplicity) continue;
      result.diagnostics.directed_edges += multiplicity;
      ++result.diagnostics.undirected_edges;
      const double d = distance(ids[i], ids[j]);
      if (!std::isfinite(d) || d < 0) throw std::invalid_argument("locality split: invalid distance");
      distances[i * n + j] = distances[j * n + i] = d;
      ++result.diagnostics.distance_evals;
      edge_min = std::min(edge_min, distances[i * n + j]);
      edge_max = std::max(edge_max, distances[i * n + j]);
    }
  }
  const bool has_edges = result.diagnostics.undirected_edges != 0;
  result.diagnostics.no_edge_fallback = !has_edges;
  const double span = has_edges ? edge_max - edge_min : 0;
  const double span_tol = has_edges ? 1e-12 * std::max(1.0, std::max(std::abs(edge_min), std::abs(edge_max))) : 0;
  std::vector<double> weights(n * n, 0.0), degree(n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      const unsigned multiplicity = directed[i * n + j] + directed[j * n + i];
      if (!multiplicity) continue;
      // A small positive floor preserves the longest actual edge. Equal
      // distances retain all actual links with equal similarity.
      const double similarity = span > span_tol ?
          std::max(1.0 / 1024.0, 1.0 - (distances[i * n + j] - edge_min) / span) : 1.0;
      const double weight = multiplicity * similarity;
      weights[i * n + j] = weights[j * n + i] = weight;
      degree[i] += weight;
      degree[j] += weight;
    }
  }
  for (double d : degree) result.diagnostics.zero_degree_nodes += (d == 0.0);

  std::vector<idx_t> offsets(n + 1, 0), neighbors, edge_weights, parts(n, 0);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (weights[i * n + j] <= 0) continue;
      neighbors.push_back(static_cast<idx_t>(j));
      edge_weights.push_back(std::max<idx_t>(1, std::llround(weights[i * n + j] * 1000000.0)));
    }
    offsets[i + 1] = static_cast<idx_t>(neighbors.size());
  }
  idx_t count = n, constraints = 1, nparts = 2, objective = 0, dummy = 0;
  idx_t options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(options);
  options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
  options[METIS_OPTION_NUMBERING] = 0;
  options[METIS_OPTION_SEED] = 42;
  options[METIS_OPTION_NCUTS] = 1;
  options[METIS_OPTION_NITER] = max_iterations;
  options[METIS_OPTION_UFACTOR] = 30;
  options[METIS_OPTION_IPTYPE] = METIS_IPTYPE_GROW;
  real_t targets[2] = {0.5, 0.5}, imbalance[1] = {1.03};
  const int status = METIS_PartGraphRecursive(&count, &constraints, offsets.data(),
      neighbors.empty() ? &dummy : neighbors.data(), nullptr, nullptr,
      edge_weights.empty() ? nullptr : edge_weights.data(), &nparts, targets,
      imbalance, options, &objective, parts.data());
  if (status != METIS_OK) throw std::runtime_error("locality split: METIS failed");
  size_t sizes[2] = {0, 0};
  for (idx_t part : parts) {
    if (part != 0 && part != 1) throw std::runtime_error("locality split: invalid METIS label");
    ++sizes[part];
    result.group.push_back(static_cast<uint8_t>(part));
  }
  if (!sizes[0] || !sizes[1] || sizes[0] > capacity || sizes[1] > capacity)
    throw std::runtime_error("locality split: METIS violated page capacity");
  result.diagnostics.initializations = 1;
  return result;
}

}  // namespace locality_split
}  // namespace pipeann
