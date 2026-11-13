#pragma once

#include <cstddef>
#include <cstdint>

#include <initializer_list>
#include <optional>
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

/// Compiler PBQP costs are small integers; sums saturate and totals use CostSum.
using Cost    = std::int16_t;
using CostSum = std::int64_t;

/// PBQP vectors and matrices are small, so they live inline.
using Costs     = ankerl::svector<Cost, 4>;
using Adjacency = ankerl::svector<EdgeId, 4>;
/// One option per live node, indexed by node id.
using Choices = ankerl::svector<OptionIndex, 8>;

/// A PBQP problem graph.
///
/// Every node carries a cost vector with one entry per option. Every edge
/// carries a cost matrix indexed by the options of its endpoints:
///
///     costs[src_option * option_count(dst) + dst_option]
///
/// The total cost of an assignment `x` is
///
///     sum_v c_v(x_v) + sum_{uv} M_uv(x_u, x_v)
///
/// Node and edge ids are stable: removing an element leaves a hole that is
/// skipped by nodes()/edges() but never reused. A node holds at most one edge
/// to any other node, so parallel edges have to be merged by summing their
/// matrices first.
class Graph {
public:
    struct Node {
        Costs costs;     // costs[option]
        Adjacency edges; // incident live edges, no duplicates
        bool alive = true;
    };

    struct Edge {
        NodeId src = 0;
        NodeId dst = 0;
        Costs costs; // row-major, src option is the row
        bool alive = true;
    };

    /// Add a node with the given cost vector. The vector must not be empty.
    NodeId add_node(std::span<const Cost> costs);
    NodeId add_node(std::initializer_list<Cost> costs) {
        return add_node(std::span<const Cost>{costs.begin(), costs.size()});
    }

    /// Link `u` and `v`. `costs` must hold exactly
    /// option_count(u) * option_count(v) entries, row-major with `u` as the
    /// row index. Self edges and duplicate edges are rejected.
    EdgeId link(NodeId u, NodeId v, std::span<const Cost> costs);
    EdgeId link(NodeId u, NodeId v, std::initializer_list<Cost> costs) {
        return link(u, v, std::span<const Cost>{costs.begin(), costs.size()});
    }

    void remove_edge(EdgeId e);
    void remove_node(NodeId v);

    [[nodiscard]] bool has_node(NodeId v) const;
    [[nodiscard]] bool has_edge(EdgeId e) const;

    [[nodiscard]] Node& node(NodeId v);
    [[nodiscard]] const Node& node(NodeId v) const;
    [[nodiscard]] Edge& edge(EdgeId e);
    [[nodiscard]] const Edge& edge(EdgeId e) const;

    [[nodiscard]] size_t option_count(NodeId v) const { return node(v).costs.size(); }

    /// Matrix entry of `e`, addressed as (src option, dst option).
    [[nodiscard]] Cost edge_cost(EdgeId e, OptionIndex src_option, OptionIndex dst_option) const;

    /// The edge between `u` and `v`, if one exists.
    [[nodiscard]] std::optional<EdgeId> find_edge(NodeId u, NodeId v) const;

    /// Views over the ids of live nodes and edges. The graph must outlive them.
    [[nodiscard]] auto nodes() const {
        return std::views::iota(NodeId{0}, nodes_.size())
             | std::views::filter([this](NodeId v) { return has_node(v); });
    }
    [[nodiscard]] auto edges() const {
        return std::views::iota(EdgeId{0}, edges_.size())
             | std::views::filter([this](EdgeId e) { return has_edge(e); });
    }

    [[nodiscard]] size_t node_count() const { return live_nodes_; }
    [[nodiscard]] size_t edge_count() const { return live_edges_; }

private:
    ankerl::svector<Node, 2> nodes_;
    ankerl::svector<Edge, 2> edges_;
    size_t live_nodes_ = 0;
    size_t live_edges_ = 0;
};

} // namespace pbqp
