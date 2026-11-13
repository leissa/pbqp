#pragma once

#include <cstddef>

#include <initializer_list>
#include <optional>
#include <ranges>
#include <span>

#include <ankerl/svector.h>

namespace pbqp {

/// Identifier of a node in a Graph. Ids are stable for the lifetime of the graph.
using NodeId = std::size_t;
/// Identifier of an edge in a Graph.
using EdgeId = std::size_t;
/// Index of an option within a node's cost vector.
using OptionIndex = std::size_t;

/// PBQP vectors and matrices are small, so they live inline.
using Costs     = ankerl::svector<double, 4>;
using Adjacency = ankerl::svector<EdgeId, 4>;
/// One option per live node, indexed by node id.
using Choices = ankerl::svector<OptionIndex, 8>;

/// A PBQP problem graph.
///
/// Every node carries a cost vector with one entry per option. Every edge
/// carries a cost matrix indexed by the options of its endpoints:
///
///     costs[source_option * option_count(target) + target_option]
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
        NodeId source = 0;
        NodeId target = 0;
        Costs costs; // row-major, source option is the row
        bool alive = true;
    };

    /// Add a node with the given cost vector. The vector must not be empty.
    auto add_node(std::span<const double> costs) -> NodeId;
    auto add_node(std::initializer_list<double> costs) -> NodeId {
        return add_node(std::span<const double>{costs.begin(), costs.size()});
    }

    /// Add an edge between `u` and `v`. `costs` must hold exactly
    /// option_count(u) * option_count(v) entries, row-major with `u` as the
    /// row index. Self edges and duplicate edges are rejected.
    auto add_edge(NodeId u, NodeId v, std::span<const double> costs) -> EdgeId;
    auto add_edge(NodeId u, NodeId v, std::initializer_list<double> costs) -> EdgeId {
        return add_edge(u, v, std::span<const double>{costs.begin(), costs.size()});
    }

    void remove_edge(EdgeId e);
    void remove_node(NodeId v);

    [[nodiscard]] auto has_node(NodeId v) const -> bool;
    [[nodiscard]] auto has_edge(EdgeId e) const -> bool;

    [[nodiscard]] auto node(NodeId v) -> Node&;
    [[nodiscard]] auto node(NodeId v) const -> const Node&;
    [[nodiscard]] auto edge(EdgeId e) -> Edge&;
    [[nodiscard]] auto edge(EdgeId e) const -> const Edge&;

    [[nodiscard]] auto option_count(NodeId v) const -> std::size_t { return node(v).costs.size(); }

    /// Matrix entry of `e`, addressed as (source option, target option).
    [[nodiscard]] auto edge_cost(EdgeId e, OptionIndex source_option, OptionIndex target_option) const -> double;

    /// The edge between `u` and `v`, if one exists.
    [[nodiscard]] auto find_edge(NodeId u, NodeId v) const -> std::optional<EdgeId>;

    /// Views over the ids of live nodes and edges. The graph must outlive them.
    [[nodiscard]] auto nodes() const {
        return std::views::iota(NodeId{0}, nodes_.size())
             | std::views::filter([this](NodeId v) { return has_node(v); });
    }
    [[nodiscard]] auto edges() const {
        return std::views::iota(EdgeId{0}, edges_.size())
             | std::views::filter([this](EdgeId e) { return has_edge(e); });
    }

    [[nodiscard]] auto node_count() const -> std::size_t {
        return static_cast<std::size_t>(std::ranges::distance(nodes()));
    }
    [[nodiscard]] auto edge_count() const -> std::size_t {
        return static_cast<std::size_t>(std::ranges::distance(edges()));
    }

private:
    ankerl::svector<Node, 2> nodes_;
    ankerl::svector<Edge, 2> edges_;
};

} // namespace pbqp
