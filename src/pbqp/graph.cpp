#include "pbqp/graph.hpp"

#include <algorithm>
#include <stdexcept>

namespace pbqp {
namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

} // namespace

NodeId Graph::add_node(std::span<const Cost> costs) {
    require(!costs.empty(), "add_node: a PBQP node needs at least one option");
    nodes_.push_back(Node{
        .costs = {costs.begin(), costs.end()},
        .edges = {},
        .alive = true
    });
    ++live_nodes_;
    return nodes_.size() - 1;
}

EdgeId Graph::link(NodeId u, NodeId v, std::span<const Cost> costs) {
    require(has_node(u) && has_node(v), "link: unknown node");
    require(u != v, "link: self edges are not supported, fold them into the node cost vector instead");
    require(!find_edge(u, v), "link: an edge between these nodes already exists, merge the matrices instead");
    require(costs.size() == option_count(u) * option_count(v), "link: cost matrix has the wrong size");

    edges_.push_back({
        .src   = u,
        .dst   = v,
        .costs = {costs.begin(), costs.end()},
        .alive = true
    });
    const EdgeId e = edges_.size() - 1;
    nodes_[u].edges.push_back(e);
    nodes_[v].edges.push_back(e);
    ++live_edges_;
    return e;
}

void Graph::remove_edge(EdgeId e) {
    require(has_edge(e), "remove_edge: unknown edge");

    Edge& removed = edges_[e];
    removed.alive = false;
    --live_edges_;
    for (const NodeId endpoint : {removed.src, removed.dst}) {
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
    --live_nodes_;
}

bool Graph::has_node(NodeId v) const { return v < nodes_.size() && nodes_[v].alive; }

bool Graph::has_edge(EdgeId e) const { return e < edges_.size() && edges_[e].alive; }

Graph::Node& Graph::node(NodeId v) {
    require(has_node(v), "node: unknown or removed node");
    return nodes_[v];
}

const Graph::Node& Graph::node(NodeId v) const {
    require(has_node(v), "node: unknown or removed node");
    return nodes_[v];
}

Graph::Edge& Graph::edge(EdgeId e) {
    require(has_edge(e), "edge: unknown or removed edge");
    return edges_[e];
}

const Graph::Edge& Graph::edge(EdgeId e) const {
    require(has_edge(e), "edge: unknown or removed edge");
    return edges_[e];
}

Cost Graph::edge_cost(EdgeId e, OptionIndex src_option, OptionIndex dst_option) const {
    const Edge& edge = this->edge(e);
    return edge.costs.at(src_option * option_count(edge.dst) + dst_option);
}

std::optional<EdgeId> Graph::find_edge(NodeId u, NodeId v) const {
    require(has_node(u) && has_node(v), "find_edge: unknown node");
    for (const EdgeId e : nodes_[u].edges) {
        const Edge& candidate = edges_[e];
        if (candidate.src == v || candidate.dst == v) return e;
    }
    return std::nullopt;
}

} // namespace pbqp
