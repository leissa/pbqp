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
    double cost = 0.0;
};

/// Cost of `choices` under `graph`: cost vector entries for nodes, matrix
/// entries for edges. Throws std::invalid_argument if `choices` is too short
/// or contains an out-of-range option.
[[nodiscard]] auto evaluate(const Graph& graph, std::span<const OptionIndex> choices) -> double;

/// Heuristic PBQP solver.
///
/// Repeatedly reduces the graph - degree 0 (pick the cheapest option),
/// degree 1 (RI) and degree 2 (RII) - and falls back to RN for higher-degree
/// nodes, then reconstructs a solution in reverse reduction order. RI and RII
/// are exact, RN is a heuristic, so the result is exact on trees.
[[nodiscard]] auto solve(const Graph& graph) -> Solution;

/// Brute-force optimum, useful to cross-check small instances.
/// Throws std::runtime_error if there are more than `max_combinations`
/// candidate assignments.
[[nodiscard]] auto solve_exhaustive(const Graph& graph, std::size_t max_combinations = 10'000'000) -> Solution;

} // namespace pbqp
