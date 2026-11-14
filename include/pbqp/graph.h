#pragma once

#include <cstddef>
#include <cstdint>

#include <initializer_list>
#include <limits>
#include <ranges>
#include <span>

#include <ankerl/svector.h>

namespace pbqp {

/// Identifier of a node in a Graph. Ids are stable for the lifetime of the graph.
using NodeId = size_t;
/// Identifier of an edge in a Graph.
using EdgeId = size_t;
/// Index of an option within a node's cost vector.
using OptionIndex = size_t;

/// Compiler PBQP costs are small integers; totals use CostSum.
using Cost    = std::int32_t;
using CostSum = std::int64_t;

/// Forbids an option or option pair; every total that contains it is kInfiniteSum.
inline constexpr Cost kInfinity       = std::numeric_limits<Cost>::max();
inline constexpr CostSum kInfiniteSum = std::numeric_limits<CostSum>::max();

/// PBQP vectors and matrices are small, so they live inline.
using Costs = ankerl::svector<Cost, 4>;
using Edges = ankerl::svector<EdgeId, 4>;
/// One option per live node, indexed by node id.
using Choices = ankerl::svector<OptionIndex, 8>;

/// A PBQP problem graph.
///
/// Every node carries a cost vector with one entry per option. Every edge
/// carries a cost matrix indexed by the options of its endpoints:
///
///     costs[src_option * num_options(dst) + dst_option]
///
/// The total cost of an assignment `x` is
///
///     sum_n c_n(x_n) + sum_{mn} M_mn(x_m, x_n)
///
/// Node and edge ids are stable: removing an element leaves a hole that is
/// skipped by nodes()/edges() but never reused. A node holds at most one edge
/// to any other node, so parallel edges have to be merged by summing their
/// matrices first.
class Graph {
public:
    static constexpr EdgeId npos = static_cast<EdgeId>(-1);

    /// Add a node with the given cost vector. The vector must not be empty.
    NodeId add(std::span<const Cost> costs);
    NodeId add(std::initializer_list<Cost> costs) { return add(std::span<const Cost>{costs.begin(), costs.size()}); }

    /// Link `m` and `n` with a row-major num_options(m) x num_options(n) matrix.
    /// Requires distinct nodes that are not linked yet.
    EdgeId link(NodeId m, NodeId n, std::span<const Cost> costs);
    EdgeId link(NodeId m, NodeId n, std::initializer_list<Cost> costs) {
        return link(m, n, std::span<const Cost>{costs.begin(), costs.size()});
    }

    void unlink(EdgeId e);
    void remove(NodeId n);

    [[nodiscard]] bool has_node(NodeId n) const;
    [[nodiscard]] bool has_edge(EdgeId e) const;

    [[nodiscard]] std::span<const Cost> node_costs(NodeId n) const;
    [[nodiscard]] std::span<const EdgeId> incident_edges(NodeId n) const;
    [[nodiscard]] size_t num_options(NodeId n) const { return node_costs(n).size(); }

    struct Endpoints {
        NodeId src = 0;
        NodeId dst = 0;
    };

    [[nodiscard]] Endpoints edge(EdgeId e) const;
    /// Row-major with the src option as the row.
    [[nodiscard]] std::span<const Cost> edge_costs(EdgeId e) const;

    /// Replace a cost vector; the option count must not change.
    void set_node_costs(NodeId n, std::span<const Cost> costs);
    /// Replace a cost matrix; the size must not change.
    void set_edge_costs(EdgeId e, std::span<const Cost> costs);

    /// Matrix entry of `e`, addressed as (src option, dst option).
    [[nodiscard]] Cost edge_cost(EdgeId e, OptionIndex src_option, OptionIndex dst_option) const;

    /// The edge between `m` and `n`, or npos.
    [[nodiscard]] EdgeId find_edge(NodeId m, NodeId n) const;

    /// Views over the ids of live nodes and edges. The graph must outlive them.
    [[nodiscard]] auto nodes() const {
        return std::views::iota(NodeId{0}, nodes_.size())
             | std::views::filter([this](NodeId n) { return has_node(n); });
    }
    [[nodiscard]] auto edges() const {
        return std::views::iota(EdgeId{0}, edges_.size())
             | std::views::filter([this](EdgeId e) { return has_edge(e); });
    }

    [[nodiscard]] size_t num_nodes() const { return live_nodes_; }
    [[nodiscard]] size_t num_edges() const { return live_edges_; }

private:
    struct Node {
        Costs costs;
        Edges edges; // incident live edges, no duplicates
        bool alive = true;
    };

    struct Edge {
        NodeId src = 0;
        NodeId dst = 0;
        Costs costs;
        bool alive = true;
    };

    [[nodiscard]] const Node& checked_node(NodeId n) const;
    [[nodiscard]] const Edge& checked_edge(EdgeId e) const;

    ankerl::svector<Node, 2> nodes_;
    ankerl::svector<Edge, 2> edges_;
    size_t live_nodes_ = 0;
    size_t live_edges_ = 0;
};

} // namespace pbqp
