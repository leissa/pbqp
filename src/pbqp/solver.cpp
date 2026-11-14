#include "pbqp/solver.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <queue>
#include <thread>
#include <vector>

#if defined(__SANITIZE_THREAD__)
#    define PBQP_TSAN 1
#elif defined(__has_feature)
#    if __has_feature(thread_sanitizer)
#        define PBQP_TSAN 1
#    endif
#endif

// ifunc resolvers run during relocation, before the ThreadSanitizer runtime exists.
#if defined(__x86_64__) && defined(__ELF__) && (defined(__GNUC__) || defined(__clang__)) && !defined(PBQP_TSAN)
#    define PBQP_DISPATCH __attribute__((target_clones("arch=x86-64-v4", "avx2", "default")))
#else
#    define PBQP_DISPATCH
#endif

namespace pbqp {
namespace {

using Vector = Costs;
using Matrix = Costs;

CostSum widen(Cost cost) { return cost == kInfinity ? kInfiniteSum : cost; }

CostSum add_sums(CostSum lhs, CostSum rhs) {
    return lhs == kInfiniteSum || rhs == kInfiniteSum ? kInfiniteSum : lhs + rhs;
}

/// Only valid for normalized, hence non-negative, costs: their sum fits into 32 unsigned bits.
Cost add_costs(Cost lhs, Cost rhs) {
    const auto sum = static_cast<std::uint32_t>(lhs) + static_cast<std::uint32_t>(rhs);
    return static_cast<Cost>(std::min(sum, static_cast<std::uint32_t>(kInfinity)));
}

/// Any sign; kInfinity absorbs.
Cost add_signed(Cost lhs, Cost rhs) {
    if (lhs == kInfinity || rhs == kInfinity) return kInfinity;
    const CostSum sum = CostSum{lhs} + rhs;
    return static_cast<Cost>(std::clamp<CostSum>(sum, std::numeric_limits<Cost>::min(), kInfinity));
}

/// Removes `minimum` from `value`; an infinite minimum means the whole row was moved elsewhere.
Cost subtract(Cost value, Cost minimum) {
    if (minimum == kInfinity) return 0;
    if (value == kInfinity) return kInfinity;
    return static_cast<Cost>(std::min<CostSum>(CostSum{value} - minimum, kInfinity));
}

PBQP_DISPATCH void add_into(Cost* __restrict dst, const Cost* __restrict src, size_t count) {
    for (size_t i = 0; i < count; ++i)
        dst[i] = add_costs(dst[i], src[i]);
}

PBQP_DISPATCH void fold_row(Cost* __restrict dst, const Cost* __restrict row, Cost cost, size_t count) {
    for (size_t i = 0; i < count; ++i)
        dst[i] = std::min(dst[i], add_costs(cost, row[i]));
}

/// Matrix of an edge oriented (reduced option, neighbour option).
struct TrimmedEdge {
    NodeId neighbour         = 0;
    size_t neighbour_options = 0;
    Matrix matrix;
};

struct Reduction {
    NodeId node = 0;
    Vector costs;
    ankerl::svector<TrimmedEdge, 4> edges;
};

Matrix transpose(const Matrix& in, size_t rows, size_t cols) {
    Matrix out(rows * cols);
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c)
            out[r * cols + c] = in[c * rows + r];
    return out;
}

/// Normalized working copy of a Graph with degree-bucketed worklists.
class Reducer {
public:
    Reducer(const Graph& graph, std::span<const size_t> rn_priority)
        : rn_priority_(rn_priority) {
        for (const auto n : graph.nodes())
            nodes_.resize(std::max(nodes_.size(), n + 1));
        assert((rn_priority.empty() || rn_priority.size() >= nodes_.size())
               && "solve: rn_priority does not cover every node");
        for (const auto n : graph.nodes()) {
            const auto costs = graph.node_costs(n);
            nodes_[n].costs.assign(costs.begin(), costs.end());
            nodes_[n].original.resize(costs.size());
            nodes_[n].original_options = costs.size();
            std::ranges::iota(nodes_[n].original, OptionIndex{0});
            nodes_[n].alive = true;
        }
        for (const auto e : graph.edges()) {
            const auto [src, dst] = graph.edge(e);
            const auto costs      = graph.edge_costs(e);
            link(src, dst, Matrix(costs.begin(), costs.end()));
        }
        for (EdgeId e = 0; e < edges_.size(); ++e)
            normalize_edge(e);
        for (const auto n : graph.nodes()) {
            normalize_node(n);
            touch(n);
        }
    }

    [[nodiscard]] size_t slots() const { return nodes_.size(); }
    [[nodiscard]] const Statistics& statistics() const { return statistics_; }

    /// Exact reductions first; RN candidates are ordered by priority, degree, then rn_score.
    NodeId next() {
        while (!dirty_.empty()) {
            const NodeId n = dirty_.back();
            dirty_.pop_back();
            nodes_[n].dirty = false;
            if (nodes_[n].alive) prune(n);
        }
        while (!low_.empty()) {
            const NodeId n = low_.back();
            low_.pop_back();
            if (nodes_[n].alive) return n;
        }
        while (!high_.empty()) {
            const Candidate candidate = high_.top();
            high_.pop();
            const Node& node = nodes_[candidate.node];
            if (node.alive && node.version == candidate.version) return candidate.node;
        }
        return Graph::npos;
    }

    void reduce(NodeId n, std::vector<Reduction>& stack) {
        if (degree(n) > 2 && options(n) > 1) {
            bool merged = false;
            while (merge_neighbour(n, stack))
                merged = true;
            if (merged) {
                touch(n);
                return;
            }
        }

        Reduction reduction = snapshot(n);
        switch (reduction.edges.size()) {
            case 0: ++statistics_.r0; break;
            case 1:
                ++statistics_.r1;
                fold_into_neighbour(reduction);
                break;
            case 2:
                ++statistics_.r2;
                contract(reduction);
                break;
            default:
                ++(options(n) == 1 ? statistics_.single_option : statistics_.rn);
                fold_locally_best_option(reduction);
                break;
        }
        stack.emplace_back(restore_original_options(std::move(reduction)));
        remove_node(n);
    }

private:
    struct Node {
        Vector costs;
        ankerl::svector<OptionIndex, 4> original; // original[working option]
        Adjacency edges;
        size_t original_options = 0;
        size_t version          = 0;
        bool alive              = false;
        bool low                = false;
        bool dirty              = false;
    };

    struct Edge {
        NodeId src = 0;
        NodeId dst = 0;
        Matrix costs;
    };

    struct Candidate {
        size_t urgency = 0; // inverted priority, so that the min-heap pops the highest priority first
        size_t degree  = 0;
        double score   = 0.0;
        NodeId node    = 0;
        size_t version = 0;

        auto operator<=>(const Candidate&) const = default;
    };

    [[nodiscard]] size_t degree(NodeId n) const { return nodes_[n].edges.size(); }
    [[nodiscard]] size_t options(NodeId n) const { return nodes_[n].costs.size(); }

    /// Briggs-style RN heuristic: small values mean "cheap to decide late".
    [[nodiscard]] double rn_score(NodeId n) const {
        const auto [low, high] = std::ranges::minmax(nodes_[n].costs);
        return static_cast<double>(high - low) / static_cast<double>(std::max<size_t>(1, degree(n)));
    }

    /// Single-option nodes are reduced exactly at any degree, so they belong on the low worklist.
    void touch(NodeId n) {
        Node& node = nodes_[n];
        ++node.version;
        if (!node.dirty) dirty_.emplace_back(n);
        node.dirty = true;
        if (degree(n) <= 2 || options(n) == 1) {
            if (!node.low) low_.emplace_back(n);
            node.low = true;
        } else {
            const size_t priority = rn_priority_.empty() ? 0 : rn_priority_[n];
            high_.push(Candidate{.urgency = ~priority,
                                 .degree  = degree(n),
                                 .score   = rn_score(n),
                                 .node    = n,
                                 .version = node.version});
        }
    }

    void normalize_node(NodeId n) {
        Vector& costs      = nodes_[n].costs;
        const Cost minimum = std::ranges::min(costs);
        if (minimum == 0 || minimum == kInfinity) return;
        for (auto& cost : costs)
            cost = subtract(cost, minimum);
    }

    /// Moves row and column minima into the node vectors and drops the edge if nothing is left.
    void normalize_edge(EdgeId e) {
        Edge& edge        = edges_[e];
        Vector& src_costs = nodes_[edge.src].costs;
        Vector& dst_costs = nodes_[edge.dst].costs;
        const size_t rows = src_costs.size();
        const size_t cols = dst_costs.size();

        for (size_t r = 0; r < rows; ++r) {
            const Cost* const row = edge.costs.data() + r * cols;
            const Cost minimum    = *std::min_element(row, row + cols);
            if (minimum == 0) continue;
            src_costs[r] = add_signed(src_costs[r], minimum);
            for (size_t c = 0; c < cols; ++c)
                edge.costs[r * cols + c] = subtract(edge.costs[r * cols + c], minimum);
        }
        for (size_t c = 0; c < cols; ++c) {
            Cost minimum = kInfinity;
            for (size_t r = 0; r < rows; ++r)
                minimum = std::min(minimum, edge.costs[r * cols + c]);
            if (minimum == 0) continue;
            dst_costs[c] = add_signed(dst_costs[c], minimum);
            for (size_t r = 0; r < rows; ++r)
                edge.costs[r * cols + c] = subtract(edge.costs[r * cols + c], minimum);
        }
        if (std::ranges::all_of(edge.costs, [](Cost cost) { return cost == 0; })) {
            erase(nodes_[edge.src].edges, e);
            erase(nodes_[edge.dst].edges, e);
            ++statistics_.dropped_edges;
        }
    }

    [[nodiscard]] NodeId other_end(EdgeId e, NodeId n) const {
        return edges_[e].src == n ? edges_[e].dst : edges_[e].src;
    }

    void erase_option(NodeId n, OptionIndex a) {
        Node& node       = nodes_[n];
        const size_t own = options(n);
        for (const auto e : node.edges) {
            Edge& edge           = edges_[e];
            const size_t other   = options(other_end(e, n));
            const bool is_src    = edge.src == n;
            const size_t cols    = is_src ? other : own;
            const size_t rows    = is_src ? own : other;
            const auto is_erased = [&](size_t index) { return is_src ? index / cols == a : index % cols == a; };

            size_t kept = 0;
            for (size_t index = 0; index < rows * cols; ++index)
                if (!is_erased(index)) edge.costs[kept++] = edge.costs[index];
            edge.costs.resize(kept);
        }
        node.costs.erase(node.costs.begin() + static_cast<std::ptrdiff_t>(a));
        node.original.erase(node.original.begin() + static_cast<std::ptrdiff_t>(a));
    }

    /// Deletes forbidden options, always keeping at least one.
    void prune(NodeId n) {
        bool changed = false;
        for (OptionIndex a = 0; a < options(n) && options(n) > 1;) {
            if (nodes_[n].costs[a] == kInfinity) {
                erase_option(n, a);
                ++statistics_.forbidden_options;
                changed = true;
            } else {
                ++a;
            }
        }
        if (!changed) return;

        const Adjacency incident = nodes_[n].edges;
        for (const auto e : incident)
            normalize_edge(e);
        normalize_node(n);
        touch(n);
        for (const auto e : incident) {
            normalize_node(other_end(e, n));
            touch(other_end(e, n));
        }
    }

    /// Re-indexes a reduction by original options; deleted options become forbidden.
    [[nodiscard]] Reduction restore_original_options(Reduction reduction) const {
        const auto unchanged = [this](NodeId n) { return nodes_[n].original.size() == nodes_[n].original_options; };
        if (unchanged(reduction.node)
            && std::ranges::all_of(reduction.edges, [&](const TrimmedEdge& edge) { return unchanged(edge.neighbour); }))
            return reduction;

        const Node& node = nodes_[reduction.node];
        Reduction restored{.node = reduction.node, .costs = Vector(node.original_options, kInfinity), .edges = {}};
        for (size_t x = 0; x < node.original.size(); ++x)
            restored.costs[node.original[x]] = reduction.costs[x];

        for (const auto& edge : reduction.edges) {
            const Node& neighbour = nodes_[edge.neighbour];
            TrimmedEdge expanded{.neighbour         = edge.neighbour,
                                 .neighbour_options = neighbour.original_options,
                                 .matrix = Matrix(node.original_options * neighbour.original_options, kInfinity)};
            for (size_t x = 0; x < node.original.size(); ++x)
                for (size_t y = 0; y < neighbour.original.size(); ++y)
                    expanded.matrix[node.original[x] * neighbour.original_options + neighbour.original[y]]
                        = edge.matrix[x * edge.neighbour_options + y];
            restored.edges.emplace_back(std::move(expanded));
        }
        return restored;
    }

    void link(NodeId m, NodeId n, Matrix costs) {
        edges_.emplace_back(Edge{.src = m, .dst = n, .costs = std::move(costs)});
        nodes_[m].edges.emplace_back(edges_.size() - 1);
        nodes_[n].edges.emplace_back(edges_.size() - 1);
    }

    [[nodiscard]] EdgeId find_edge(NodeId m, NodeId n) const {
        if (degree(n) < degree(m)) std::swap(m, n);
        for (const auto e : nodes_[m].edges)
            if (edges_[e].src == n || edges_[e].dst == n) return e;
        return Graph::npos;
    }

    void remove_node(NodeId n) {
        Node& node = nodes_[n];
        for (const auto e : node.edges) {
            const NodeId other = edges_[e].src == n ? edges_[e].dst : edges_[e].src;
            erase(nodes_[other].edges, e);
        }
        const Adjacency incident = std::move(node.edges);
        node.edges.clear();
        node.alive = false;
        for (const auto e : incident)
            touch(edges_[e].src == n ? edges_[e].dst : edges_[e].src);
    }

    [[nodiscard]] TrimmedEdge trim_edge(EdgeId e, NodeId from) const {
        const Edge& edge   = edges_[e];
        const bool is_src  = edge.src == from;
        const NodeId other = is_src ? edge.dst : edge.src;
        const size_t rows  = options(from);
        const size_t cols  = options(other);
        return TrimmedEdge{
            .neighbour         = other,
            .neighbour_options = cols,
            .matrix            = is_src ? edge.costs : transpose(edge.costs, rows, cols),
        };
    }

    [[nodiscard]] Reduction snapshot(NodeId n) const {
        Reduction reduction{.node = n, .costs = nodes_[n].costs, .edges = {}};
        reduction.edges.reserve(degree(n));
        for (const auto e : nodes_[n].edges)
            reduction.edges.emplace_back(trim_edge(e, n));
        return reduction;
    }

    /// RI: c'_n(y) = c_n(y) + min_x ( c(x) + M(x, y) )
    void fold_into_neighbour(const Reduction& reduction) {
        const TrimmedEdge& edge = reduction.edges.front();
        const size_t y_dim      = edge.neighbour_options;

        Vector best(y_dim, kInfinity);
        for (size_t x = 0; x < reduction.costs.size(); ++x)
            fold_row(best.data(), edge.matrix.data() + x * y_dim, reduction.costs[x], y_dim);
        add_into(nodes_[edge.neighbour].costs.data(), best.data(), y_dim);
        normalize_node(edge.neighbour);
    }

    /// RII: M'_mn(i, j) = min_x ( c(x) + M_m(x, i) + M_n(x, j) )
    void contract(const Reduction& reduction) {
        const TrimmedEdge& to_m = reduction.edges[0];
        const TrimmedEdge& to_n = reduction.edges[1];
        const size_t m_dim      = to_m.neighbour_options;
        const size_t n_dim      = to_n.neighbour_options;

        Matrix merged(m_dim * n_dim, kInfinity);
        for (size_t x = 0; x < reduction.costs.size(); ++x) {
            const Cost* const n_row = to_n.matrix.data() + x * n_dim;
            for (size_t i = 0; i < m_dim; ++i)
                fold_row(merged.data() + i * n_dim, n_row, add_costs(reduction.costs[x], to_m.matrix[x * m_dim + i]),
                         n_dim);
        }

        add_to_edge(to_m.neighbour, to_n.neighbour, std::move(merged));
    }

    /// Adds `matrix` (oriented (m, n)) to the edge between m and n, creating it if necessary.
    void add_to_edge(NodeId m, NodeId n, Matrix matrix) {
        EdgeId e = find_edge(m, n);
        if (e != Graph::npos) {
            Edge& edge = edges_[e];
            if (edge.src != m) matrix = transpose(matrix, options(n), options(m));
            add_into(edge.costs.data(), matrix.data(), matrix.size());
        } else {
            e = edges_.size();
            link(m, n, std::move(matrix));
        }
        normalize_edge(e);
        normalize_node(m);
        normalize_node(n);
    }

    /// The option of `m` that each option of `n` forces, or nothing if some option of `n` leaves `m` a choice.
    [[nodiscard]] std::optional<ankerl::svector<OptionIndex, 4>> implied_options(EdgeId e, NodeId n) const {
        const TrimmedEdge edge = trim_edge(e, n);
        const Vector& m_costs  = nodes_[edge.neighbour].costs;
        ankerl::svector<OptionIndex, 4> implied(options(n), Graph::npos);
        for (OptionIndex i = 0; i < options(n); ++i) {
            if (nodes_[n].costs[i] == kInfinity) continue;
            for (OptionIndex j = 0; j < edge.neighbour_options; ++j) {
                if (add_costs(edge.matrix[i * edge.neighbour_options + j], m_costs[j]) == kInfinity) continue;
                if (implied[i] != Graph::npos) return std::nullopt;
                implied[i] = j;
            }
        }
        return implied;
    }

    /// RM (Buchwald et al. 2011): moves the other edges of a neighbour whose option `n` determines onto `n`.
    bool merge_neighbour(NodeId n, std::vector<Reduction>& stack) {
        for (const auto e : nodes_[n].edges) {
            const NodeId m = other_end(e, n);
            if (degree(m) < 2) continue;
            const auto implied = implied_options(e, n);
            if (!implied) continue;

            const Adjacency moved = nodes_[m].edges;
            for (const auto f : moved) {
                if (f == e) continue;
                const NodeId w         = other_end(f, m);
                const TrimmedEdge to_w = trim_edge(f, m);
                const size_t w_options = to_w.neighbour_options;
                Matrix delta(options(n) * w_options, 0);
                for (OptionIndex i = 0; i < options(n); ++i)
                    if ((*implied)[i] != Graph::npos)
                        std::ranges::copy_n(to_w.matrix.data() + (*implied)[i] * w_options,
                                            static_cast<std::ptrdiff_t>(w_options), delta.data() + i * w_options);
                erase(nodes_[m].edges, f);
                erase(nodes_[w].edges, f);
                add_to_edge(n, w, std::move(delta));
                touch(w);
            }
            ++statistics_.rm;
            reduce(m, stack);
            return true;
        }
        return false;
    }

    /// RN: commit to the locally best option and charge its matrix rows to the neighbours.
    void fold_locally_best_option(const Reduction& reduction) {
        OptionIndex best_option = 0;
        CostSum best_total      = kInfiniteSum;
        for (OptionIndex x = 0; x < reduction.costs.size(); ++x) {
            CostSum total = widen(reduction.costs[x]);
            for (const auto& edge : reduction.edges) {
                const Vector& neighbour_costs = nodes_[edge.neighbour].costs;
                CostSum cheapest              = kInfiniteSum;
                for (size_t y = 0; y < edge.neighbour_options; ++y)
                    cheapest = std::min(cheapest, add_sums(widen(neighbour_costs[y]),
                                                           widen(edge.matrix[x * edge.neighbour_options + y])));
                total = add_sums(total, cheapest);
            }
            if (total < best_total) {
                best_total  = total;
                best_option = x;
            }
        }
        for (const auto& edge : reduction.edges) {
            add_into(nodes_[edge.neighbour].costs.data(), edge.matrix.data() + best_option * edge.neighbour_options,
                     edge.neighbour_options);
            normalize_node(edge.neighbour);
        }
    }

    std::vector<Node> nodes_;
    std::vector<Edge> edges_;
    std::vector<NodeId> low_;
    std::vector<NodeId> dirty_;
    std::span<const size_t> rn_priority_;
    Statistics statistics_;
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>> high_;
};

/// Exact for degree <= 2 snapshots; RN nodes may still improve on their RN choice here.
OptionIndex back_substitute(const Reduction& reduction, const Choices& choices) {
    OptionIndex best   = 0;
    CostSum best_total = kInfiniteSum;

    for (OptionIndex option = 0; option < reduction.costs.size(); ++option) {
        CostSum total = widen(reduction.costs[option]);
        for (const auto& edge : reduction.edges)
            total = add_sums(total, widen(edge.matrix[option * edge.neighbour_options + choices[edge.neighbour]]));
        if (total < best_total) {
            best_total = total;
            best       = option;
        }
    }
    return best;
}

} // namespace

CostSum evaluate(const Graph& graph, std::span<const OptionIndex> choices) {
    CostSum total = 0;

    for (const auto n : graph.nodes()) {
        assert(n < choices.size() && "evaluate: choices is too short");
        const auto costs = graph.node_costs(n);
        assert(choices[n] < costs.size() && "evaluate: option index out of range");
        total = add_sums(total, widen(costs[choices[n]]));
    }
    for (const auto e : graph.edges()) {
        const auto [src, dst] = graph.edge(e);
        total = add_sums(total, widen(graph.edge_costs(e)[choices[src] * graph.option_count(dst) + choices[dst]]));
    }
    return total;
}

Solution solve(const Graph& graph, std::span<const size_t> rn_priority) {
    Reducer reducer{graph, rn_priority};
    std::vector<Reduction> stack;
    stack.reserve(graph.node_count());

    for (NodeId n = reducer.next(); n != Graph::npos; n = reducer.next())
        reducer.reduce(n, stack);

    Choices choices(reducer.slots(), 0);
    for (auto it = stack.rbegin(); it != stack.rend(); ++it)
        choices[it->node] = back_substitute(*it, choices);

    const CostSum cost = evaluate(graph, choices);
    return Solution{.choices = std::move(choices), .cost = cost, .statistics = reducer.statistics()};
}

namespace {

/// Mixed-radix layout of the assignment space; digit i belongs to nodes[i].
struct Enumeration {
    ankerl::svector<NodeId, 16> nodes;
    ankerl::svector<size_t, 16> options;
    ankerl::svector<size_t, 16> place;
    size_t slots        = 0;
    size_t combinations = 1;
};

std::optional<Enumeration> make_enumeration(const Graph& graph, size_t max_combinations) {
    Enumeration enumeration;
    for (const auto n : graph.nodes()) {
        enumeration.nodes.emplace_back(n);
        enumeration.options.emplace_back(graph.option_count(n));
    }
    if (enumeration.nodes.empty()) return enumeration;

    enumeration.slots = std::ranges::max(enumeration.nodes) + 1;
    enumeration.place.resize(enumeration.nodes.size());

    size_t combinations = 1;
    for (const auto options : enumeration.options) {
        if (combinations > max_combinations / options) return std::nullopt;
        combinations *= options;
    }
    enumeration.combinations = combinations;

    size_t weight = 1;
    for (size_t i = 0; i < enumeration.options.size(); ++i) {
        enumeration.place[i] = weight;
        weight *= enumeration.options[i];
    }
    return enumeration;
}

struct BestAssignment {
    Solution solution;
    size_t index = 0;
};

/// The smallest index wins ties so that the result does not depend on the partitioning.
BestAssignment search_range(const Graph& graph, const Enumeration& enumeration, size_t begin, size_t end) {
    Choices choices(enumeration.slots, 0);
    for (size_t i = 0; i < enumeration.nodes.size(); ++i)
        choices[enumeration.nodes[i]] = (begin / enumeration.place[i]) % enumeration.options[i];

    BestAssignment best{
        .solution = Solution{.choices = choices, .cost = kInfiniteSum, .statistics = {}},
        .index    = end
    };
    for (size_t index = begin; index < end; ++index) {
        const CostSum cost = evaluate(graph, choices);
        if (cost < best.solution.cost) {
            best.solution = Solution{.choices = choices, .cost = cost, .statistics = {}};
            best.index    = index;
        }
        for (size_t i = 0; i < enumeration.nodes.size(); ++i) {
            if (++choices[enumeration.nodes[i]] < enumeration.options[i]) break;
            choices[enumeration.nodes[i]] = 0;
        }
    }
    return best;
}

} // namespace

std::optional<Solution> solve_exhaustive(const Graph& graph, size_t max_combinations, unsigned thread_count) {
    const auto made = make_enumeration(graph, max_combinations);
    if (!made) return std::nullopt;
    const Enumeration& enumeration = *made;
    if (enumeration.nodes.empty()) return Solution{};

    unsigned requested = thread_count;
    if (requested == 0) requested = std::thread::hardware_concurrency();
    if (requested == 0) requested = 1;
    const size_t workers = std::min<size_t>(requested, enumeration.combinations);

    if (workers <= 1) return search_range(graph, enumeration, 0, enumeration.combinations).solution;

    std::vector<BestAssignment> partials(workers);
    {
        std::vector<std::jthread> threads;
        threads.reserve(workers);

        const size_t base  = enumeration.combinations / workers;
        const size_t extra = enumeration.combinations % workers;
        size_t begin       = 0;
        for (size_t i = 0; i < workers; ++i) {
            const size_t end = begin + base + (i < extra ? 1 : 0);
            threads.emplace_back([&graph, &enumeration, &partials, i, begin, end] {
                partials[i] = search_range(graph, enumeration, begin, end);
            });
            begin = end;
        }
    }

    BestAssignment best = partials[0];
    for (size_t i = 1; i < workers; ++i) {
        const BestAssignment& candidate = partials[i];
        if (candidate.solution.cost < best.solution.cost
            || (candidate.solution.cost == best.solution.cost && candidate.index < best.index))
            best = candidate;
    }
    return best.solution;
}

} // namespace pbqp
