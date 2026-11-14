#pragma once

#include <ostream>
#include <span>

#include "pbqp/graph.hpp"

namespace pbqp {

/// Write `graph` in Graphviz DOT format with cost vectors and matrices as HTML-like labels.
/// Edge matrices are titled `src × dst`, i.e. rows × columns.
/// Highlights the entries picked by `choices`, which must be empty or cover every node id.
void write_dot(std::ostream& out, const Graph& graph, std::span<const OptionIndex> choices = {});

} // namespace pbqp
