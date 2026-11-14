#include "pbqp/graph.hpp"

#include <cassert>

#include <algorithm>

namespace pbqp {

NodeId Graph::add_node(std::span<const Cost> costs) {
    assert((!costs.empty()) && "add_node: a PBQP node needs at least one option");
    nodes_.emplace_back(Node{
        .costs = {costs.begin(), costs.end()},
        .edges = {},
        .alive = true
    });
    ++live_nodes_;
    return nodes_.size() - 1;
}

EdgeId Graph::link(NodeId m, NodeId n, std::span<const Cost> costs) {
    assert((has_node(m) && has_node(n)) && "link: unknown node");
    assert((m != n) && "link: self edges are not supported, fold them into the node cost vector instead");
    assert((find_edge(m, n) == npos) && "link: an edge between these nodes already exists, merge the matrices instead");
    assert((costs.size() == option_count(m) * option_count(n)) && "link: cost matrix has the wrong size");

    edges_.emplace_back(Edge{
        .src   = m,
        .dst   = n,
        .costs = {costs.begin(), costs.end()},
        .alive = true
    });
    const EdgeId e = edges_.size() - 1;
    nodes_[m].edges.emplace_back(e);
    nodes_[n].edges.emplace_back(e);
    ++live_edges_;
    return e;
}

void Graph::remove_edge(EdgeId e) {
    assert((has_edge(e)) && "remove_edge: unknown edge");

    Edge& removed = edges_[e];
    removed.alive = false;
    --live_edges_;
    for (const auto endpoint : {removed.src, removed.dst}) {
        auto& incident = nodes_[endpoint].edges;
        erase(incident, e);
    }
}

void Graph::remove_node(NodeId n) {
    assert((has_node(n)) && "remove_node: unknown node");

    const Adjacency incident = nodes_[n].edges;
    for (const auto e : incident)
        remove_edge(e);
    nodes_[n].alive = false;
    --live_nodes_;
}

bool Graph::has_node(NodeId n) const { return n < nodes_.size() && nodes_[n].alive; }

bool Graph::has_edge(EdgeId e) const { return e < edges_.size() && edges_[e].alive; }

const Graph::Node& Graph::checked_node(NodeId n) const {
    assert((has_node(n)) && "node: unknown or removed node");
    return nodes_[n];
}

const Graph::Edge& Graph::checked_edge(EdgeId e) const {
    assert((has_edge(e)) && "edge: unknown or removed edge");
    return edges_[e];
}

std::span<const Cost> Graph::node_costs(NodeId n) const { return checked_node(n).costs; }

std::span<const EdgeId> Graph::incident_edges(NodeId n) const { return checked_node(n).edges; }

Graph::Endpoints Graph::edge(EdgeId e) const {
    const Edge& edge = checked_edge(e);
    return {.src = edge.src, .dst = edge.dst};
}

std::span<const Cost> Graph::edge_costs(EdgeId e) const { return checked_edge(e).costs; }

void Graph::set_node_costs(NodeId n, std::span<const Cost> costs) {
    assert((costs.size() == option_count(n)) && "set_node_costs: option count must not change");
    std::ranges::copy(costs, nodes_[n].costs.begin());
}

void Graph::set_edge_costs(EdgeId e, std::span<const Cost> costs) {
    assert((costs.size() == checked_edge(e).costs.size()) && "set_edge_costs: cost matrix has the wrong size");
    std::ranges::copy(costs, edges_[e].costs.begin());
}

Cost Graph::edge_cost(EdgeId e, OptionIndex src_option, OptionIndex dst_option) const {
    const Edge& edge = checked_edge(e);
    return edge.costs[src_option * option_count(edge.dst) + dst_option];
}

EdgeId Graph::find_edge(NodeId m, NodeId n) const {
    assert((has_node(m) && has_node(n)) && "find_edge: unknown node");
    for (const auto e : nodes_[m].edges) {
        const Edge& candidate = edges_[e];
        if (candidate.src == n || candidate.dst == n) return e;
    }
    return npos;
}

} // namespace pbqp
