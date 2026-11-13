#include "pbqp/graph.hpp"

#include <algorithm>
#include <stdexcept>

namespace pbqp {
namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

} // namespace

auto Graph::add_node(std::span<const double> costs) -> NodeId {
    require(!costs.empty(), "add_node: a PBQP node needs at least one option");
    nodes_.push_back(Node{
        .costs = {costs.begin(), costs.end()},
        .edges = {},
        .alive = true
    });
    return nodes_.size() - 1;
}

auto Graph::add_edge(NodeId u, NodeId v, std::span<const double> costs) -> EdgeId {
    require(has_node(u) && has_node(v), "add_edge: unknown node");
    require(u != v, "add_edge: self edges are not supported, fold them into the node cost vector instead");
    require(!find_edge(u, v), "add_edge: an edge between these nodes already exists, merge the matrices instead");
    require(costs.size() == option_count(u) * option_count(v), "add_edge: cost matrix has the wrong size");

    edges_.push_back(Edge{
        .source = u,
        .target = v,
        .costs  = {costs.begin(), costs.end()},
        .alive  = true
    });
    const EdgeId e = edges_.size() - 1;
    nodes_[u].edges.push_back(e);
    nodes_[v].edges.push_back(e);
    return e;
}

void Graph::remove_edge(EdgeId e) {
    require(has_edge(e), "remove_edge: unknown edge");

    Edge& removed = edges_[e];
    removed.alive = false;
    for (const NodeId endpoint : {removed.source, removed.target}) {
        auto& incident = nodes_[endpoint].edges;
        erase(incident, e);
    }
}

void Graph::remove_node(NodeId v) {
    require(has_node(v), "remove_node: unknown node");

    const Adjacency incident = nodes_[v].edges;
    for (const EdgeId e : incident)
        remove_edge(e);
    nodes_[v].alive = false;
}

auto Graph::has_node(NodeId v) const -> bool { return v < nodes_.size() && nodes_[v].alive; }

auto Graph::has_edge(EdgeId e) const -> bool { return e < edges_.size() && edges_[e].alive; }

auto Graph::node(NodeId v) -> Node& {
    require(has_node(v), "node: unknown or removed node");
    return nodes_[v];
}

auto Graph::node(NodeId v) const -> const Node& {
    require(has_node(v), "node: unknown or removed node");
    return nodes_[v];
}

auto Graph::edge(EdgeId e) -> Edge& {
    require(has_edge(e), "edge: unknown or removed edge");
    return edges_[e];
}

auto Graph::edge(EdgeId e) const -> const Edge& {
    require(has_edge(e), "edge: unknown or removed edge");
    return edges_[e];
}

auto Graph::edge_cost(EdgeId e, OptionIndex source_option, OptionIndex target_option) const -> double {
    const Edge& edge = this->edge(e);
    return edge.costs.at(source_option * option_count(edge.target) + target_option);
}

auto Graph::find_edge(NodeId u, NodeId v) const -> std::optional<EdgeId> {
    require(has_node(u) && has_node(v), "find_edge: unknown node");
    for (const EdgeId e : nodes_[u].edges) {
        const Edge& candidate = edges_[e];
        if (candidate.source == v || candidate.target == v) return e;
    }
    return std::nullopt;
}

} // namespace pbqp
