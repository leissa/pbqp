#pragma once

#include <cstddef>

#include <span>

#include "pbqp/graph.hpp"

namespace pbqp {

/// A complete assignment of options to nodes.
struct Solution {
    /// choices[n] is the option chosen for node `n`.
    Choices choices;
    /// Total cost of the assignment under the original graph.
    CostSum cost = 0;
};

/// All solver functions are safe to call concurrently on the same graph; a
/// Graph must not be mutated while another thread reads it.
///
/// Cost of `choices` under `graph`, or kInfiniteSum if it picks a kInfinity entry.
/// Throws std::invalid_argument if `choices` is too short or contains an out-of-range option.
[[nodiscard]] CostSum evaluate(const Graph& graph, std::span<const OptionIndex> choices);

/// Heuristic PBQP solver.
///
/// Reduces degree 0, 1 (RI) and 2 (RII) nodes exactly and higher-degree nodes with the RN heuristic.
/// Normalization keeps intermediate costs relative, so they saturate only where option costs differ by kInfinity.
/// Short of that, the result is optimal whenever no RN step is needed, e.g. on trees.
[[nodiscard]] Solution solve(const Graph& graph);

/// Brute-force optimum, useful to cross-check small instances. The search
/// space is split across `thread_count` threads (0 = hardware concurrency).
/// Throws std::runtime_error if there are more than `max_combinations`
/// candidate assignments.
[[nodiscard]] Solution
solve_exhaustive(const Graph& graph, size_t max_combinations = 10'000'000, unsigned thread_count = 0);

} // namespace pbqp
