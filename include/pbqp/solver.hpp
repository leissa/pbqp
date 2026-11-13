#pragma once

#include <cstddef>

#include <span>

#include "pbqp/graph.hpp"

namespace pbqp {

/// A complete assignment of options to nodes.
struct Solution {
    /// choices[v] is the option chosen for node `v`.
    Choices choices;
    /// Total cost of the assignment under the original graph.
    CostSum cost = 0;
};

/// All solver functions are safe to call concurrently on the same graph; a
/// Graph must not be mutated while another thread reads it.
///
/// Cost of `choices` under `graph`: cost vector entries for nodes, matrix
/// entries for edges. Throws std::invalid_argument if `choices` is too short
/// or contains an out-of-range option.
[[nodiscard]] CostSum evaluate(const Graph& graph, std::span<const OptionIndex> choices);

/// Heuristic PBQP solver.
///
/// Repeatedly reduces the graph - degree 0 (pick the cheapest option),
/// degree 1 (RI) and degree 2 (RII) - and falls back to RN for higher-degree
/// nodes, then reconstructs a solution in reverse reduction order. RI and RII
/// are exact, RN is a heuristic, so the result is exact on trees.
[[nodiscard]] Solution solve(const Graph& graph);

/// Brute-force optimum, useful to cross-check small instances. The search
/// space is split across `thread_count` threads (0 = hardware concurrency).
/// Throws std::runtime_error if there are more than `max_combinations`
/// candidate assignments.
[[nodiscard]] Solution
solve_exhaustive(const Graph& graph, size_t max_combinations = 10'000'000, unsigned thread_count = 0);

} // namespace pbqp
