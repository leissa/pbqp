#include "pbqp/solver.hpp"

#include <cstddef>

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>

namespace pbqp {
namespace {

using Vector = Costs;
using Matrix = Costs;

constexpr double kInfinity = std::numeric_limits<double>::infinity();

/// Snapshot of one edge as seen from the node that is being reduced. The
/// matrix is oriented (reduced option, neighbour option), so it can be
/// evaluated with the neighbour's final choice during back-substitution.
struct TrimmedEdge {
    NodeId neighbour              = 0;
    std::size_t neighbour_options = 0;
    Matrix matrix;
};

/// Everything needed to restore one node and to choose its option once the
/// choices of its former neighbours are known.
struct Reduction {
    NodeId node = 0;
    Vector costs;
    ankerl::svector<TrimmedEdge, 4> edges;
};

auto transpose(const Matrix& in, std::size_t rows, std::size_t cols) -> Matrix {
    Matrix out(rows * cols);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c)
            out[r * cols + c] = in[c * rows + r];
    return out;
}

/// Copy the matrix of edge `e` into (from option, neighbour option) order.
auto trim_edge(const Graph& graph, EdgeId e, NodeId from) -> TrimmedEdge {
    const Graph::Edge& edge = graph.edge(e);
    const bool is_source    = edge.source == from;
    const NodeId other      = is_source ? edge.target : edge.source;
    const std::size_t rows  = graph.option_count(from);
    const std::size_t cols  = graph.option_count(other);

    return TrimmedEdge{
        .neighbour         = other,
        .neighbour_options = cols,
        .matrix            = is_source ? edge.costs : transpose(edge.costs, cols, rows),
    };
}

/// Number of distinct neighbours of a node.
auto degree(const Graph& graph, NodeId v) -> std::size_t { return graph.node(v).edges.size(); }

/// Briggs-style RN heuristic: spread of the cost vector per incident edge.
/// Small values mean "cheap to decide late".
auto rn_score(const Graph& graph, NodeId v) -> double {
    const auto [low, high] = std::ranges::minmax(graph.node(v).costs);
    const auto incident    = static_cast<double>(std::max<std::size_t>(1, degree(graph, v)));
    return (high - low) / incident;
}

/// Pick the next node to reduce: lowest degree first, then lowest RN score.
auto select_node(const Graph& graph) -> std::optional<NodeId> {
    std::optional<NodeId> best;
    std::size_t best_degree = 0;
    double best_score       = 0.0;

    for (const NodeId v : graph.nodes()) {
        const std::size_t current_degree = degree(graph, v);
        const double current_score       = rn_score(graph, v);
        const bool better
            = !best || current_degree < best_degree || (current_degree == best_degree && current_score < best_score);
        if (better) {
            best        = v;
            best_degree = current_degree;
            best_score  = current_score;
        }
    }
    return best;
}

auto snapshot(const Graph& graph, NodeId v) -> Reduction {
    Reduction reduction{.node = v, .costs = graph.node(v).costs, .edges = {}};
    reduction.edges.reserve(degree(graph, v));
    for (const EdgeId e : graph.node(v).edges)
        reduction.edges.push_back(trim_edge(graph, e, v));
    return reduction;
}

/// R1: fold a degree-1 node into its neighbour.
///
///     c'_u(y) = c_u(y) + min_x ( c_v(x) + M(x, y) )
void fold_into_neighbour(Graph& graph, NodeId v) {
    const TrimmedEdge edge  = trim_edge(graph, graph.node(v).edges.front(), v);
    const Vector costs      = graph.node(v).costs;
    const std::size_t x_dim = costs.size();
    const std::size_t y_dim = edge.neighbour_options;

    Vector& neighbour_costs = graph.node(edge.neighbour).costs;
    for (std::size_t y = 0; y < y_dim; ++y) {
        double cheapest = kInfinity;
        for (std::size_t x = 0; x < x_dim; ++x)
            cheapest = std::min(cheapest, costs[x] + edge.matrix[x * y_dim + y]);
        neighbour_costs[y] += cheapest;
    }
}

/// Add `matrix` (oriented (u, v)) to the edge between u and v, creating it if
/// necessary. RII can create parallel edges, and parallel edges are equivalent
/// to their sum.
void add_or_merge_edge(Graph& graph, NodeId u, NodeId v, const Matrix& matrix) {
    const std::size_t cols = graph.option_count(v);

    if (const auto existing = graph.find_edge(u, v)) {
        Graph::Edge& edge           = graph.edge(*existing);
        const bool is_forward       = edge.source == u;
        const std::size_t edge_cols = graph.option_count(edge.target);

        for (std::size_t r = 0; r < graph.option_count(u); ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                const double value = matrix[r * cols + c];
                if (is_forward)
                    edge.costs[r * edge_cols + c] += value;
                else
                    edge.costs[c * edge_cols + r] += value;
            }
        }
        return;
    }
    graph.add_edge(u, v, matrix);
}

/// R2: contract a degree-2 node `v` with neighbours u and w.
///
///     M'_uw(i, j) = min_x ( c_v(x) + M_uv(x, i) + M_vw(x, j) )
void contract_node(Graph& graph, NodeId v) {
    const TrimmedEdge to_u = trim_edge(graph, graph.node(v).edges[0], v);
    const TrimmedEdge to_w = trim_edge(graph, graph.node(v).edges[1], v);
    const Vector costs     = graph.node(v).costs;

    const std::size_t u_dim = to_u.neighbour_options;
    const std::size_t w_dim = to_w.neighbour_options;
    const std::size_t x_dim = costs.size();

    Matrix merged(u_dim * w_dim, kInfinity);
    for (std::size_t i = 0; i < u_dim; ++i) {
        for (std::size_t j = 0; j < w_dim; ++j) {
            double cheapest = kInfinity;
            for (std::size_t x = 0; x < x_dim; ++x)
                cheapest = std::min(cheapest, costs[x] + to_u.matrix[x * u_dim + i] + to_w.matrix[x * w_dim + j]);
            merged[i * w_dim + j] = cheapest;
        }
    }
    add_or_merge_edge(graph, to_u.neighbour, to_w.neighbour, merged);
}

/// Choose the cheapest option for a restored node, given the choices of the
/// neighbours captured in its reduction. Exact for snapshots of degree <= 2
/// nodes, greedy otherwise.
auto back_substitute(const Reduction& reduction, const Choices& choices) -> OptionIndex {
    OptionIndex best  = 0;
    double best_total = kInfinity;

    for (OptionIndex option = 0; option < reduction.costs.size(); ++option) {
        double total = reduction.costs[option];
        for (const TrimmedEdge& edge : reduction.edges)
            total += edge.matrix[option * edge.neighbour_options + choices[edge.neighbour]];
        if (total < best_total) {
            best_total = total;
            best       = option;
        }
    }
    return best;
}

} // namespace

auto evaluate(const Graph& graph, std::span<const OptionIndex> choices) -> double {
    double total = 0.0;

    for (const NodeId v : graph.nodes()) {
        if (choices.size() <= v) throw std::invalid_argument("evaluate: choices is too short");
        const auto& costs = graph.node(v).costs;
        if (choices[v] >= costs.size()) throw std::invalid_argument("evaluate: option index out of range");
        total += costs[choices[v]];
    }
    for (const EdgeId e : graph.edges()) {
        const Graph::Edge& edge = graph.edge(e);
        total += edge.costs.at(choices[edge.source] * graph.option_count(edge.target) + choices[edge.target]);
    }
    return total;
}

auto solve(const Graph& graph) -> Solution {
    Graph working = graph;
    ankerl::svector<Reduction, 8> stack;
    stack.reserve(graph.node_count());

    while (const auto v = select_node(working)) {
        const std::size_t d = degree(working, *v);
        Reduction reduction = snapshot(working, *v);

        if (d == 1)
            fold_into_neighbour(working, *v);
        else if (d == 2)
            contract_node(working, *v);

        working.remove_node(*v);
        stack.push_back(std::move(reduction));
    }

    std::size_t slots = 0;
    for (const NodeId v : graph.nodes())
        slots = std::max(slots, v + 1);

    Choices choices(slots, 0);
    for (auto it = stack.rbegin(); it != stack.rend(); ++it)
        choices[it->node] = back_substitute(*it, choices);

    const double cost = evaluate(graph, choices);
    return Solution{.choices = std::move(choices), .cost = cost};
}

auto solve_exhaustive(const Graph& graph, std::size_t max_combinations) -> Solution {
    ankerl::svector<NodeId, 16> nodes(graph.nodes().begin(), graph.nodes().end());
    if (nodes.empty()) return Solution{};

    const std::size_t slots = std::ranges::max(nodes) + 1;
    Choices choices(slots, 0);

    std::size_t combinations = 1;
    for (const NodeId v : nodes) {
        const std::size_t options = graph.option_count(v);
        if (combinations > max_combinations / options) throw std::runtime_error("solve_exhaustive: instance too large");
        combinations *= options;
    }

    Solution best{.choices = choices, .cost = kInfinity};
    bool done = false;
    while (!done) {
        const double cost = evaluate(graph, choices);
        if (cost < best.cost) best = Solution{.choices = choices, .cost = cost};

        done = true;
        for (const NodeId v : nodes) {
            if (++choices[v] < graph.option_count(v)) {
                done = false;
                break;
            }
            choices[v] = 0;
        }
    }
    return best;
}

} // namespace pbqp
