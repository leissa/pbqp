#include "pbqp/solver.hpp"

#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <thread>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#    include <immintrin.h>
#    define PBQP_X86_DISPATCH 1
#else
#    define PBQP_X86_DISPATCH 0
#endif

#if defined(__ARM_NEON)
#    include <arm_neon.h>
#endif

namespace pbqp {
namespace {

using Vector = Costs;
using Matrix = Costs;

constexpr Cost kInfinity   = std::numeric_limits<Cost>::max();
constexpr CostSum kMaxCost = std::numeric_limits<CostSum>::max();

Cost add_saturated(Cost lhs, Cost rhs) {
    const auto sum = static_cast<std::int32_t>(lhs) + static_cast<std::int32_t>(rhs);
    return static_cast<Cost>(std::clamp<std::int32_t>(sum, std::numeric_limits<Cost>::min(), kInfinity));
}

#if PBQP_X86_DISPATCH

enum class Simd { scalar, avx2, avx512 };

/// Picks the widest kernel once; the scalar loop handles everything else.
Simd simd_level() {
    static const Simd level = [] {
        __builtin_cpu_init();
        if (__builtin_cpu_supports("avx512bw")) return Simd::avx512;
        if (__builtin_cpu_supports("avx2")) return Simd::avx2;
        return Simd::scalar;
    }();
    return level;
}

__attribute__((target("avx2"))) void add_into_avx2(Cost* __restrict dst, const Cost* __restrict src, size_t count) {
    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + i));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_adds_epi16(a, b));
    }
    for (; i < count; ++i)
        dst[i] = add_saturated(dst[i], src[i]);
}

__attribute__((target("avx512f,avx512bw"))) void
add_into_avx512(Cost* __restrict dst, const Cost* __restrict src, size_t count) {
    size_t i = 0;
    for (; i + 32 <= count; i += 32) {
        const __m512i a = _mm512_loadu_si512(dst + i);
        const __m512i b = _mm512_loadu_si512(src + i);
        _mm512_storeu_si512(dst + i, _mm512_adds_epi16(a, b));
    }
    for (; i < count; ++i)
        dst[i] = add_saturated(dst[i], src[i]);
}

__attribute__((target("avx2"))) void
fold_row_avx2(Cost* __restrict dst, const Cost* __restrict row, Cost cost, size_t count) {
    const __m256i constant = _mm256_set1_epi16(cost);
    size_t i               = 0;
    for (; i + 16 <= count; i += 16) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + i));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_min_epi16(a, _mm256_adds_epi16(constant, b)));
    }
    for (; i < count; ++i)
        dst[i] = std::min(dst[i], add_saturated(cost, row[i]));
}

__attribute__((target("avx512f,avx512bw"))) void
fold_row_avx512(Cost* __restrict dst, const Cost* __restrict row, Cost cost, size_t count) {
    const __m512i constant = _mm512_set1_epi16(cost);
    size_t i               = 0;
    for (; i + 32 <= count; i += 32) {
        const __m512i a = _mm512_loadu_si512(dst + i);
        const __m512i b = _mm512_loadu_si512(row + i);
        _mm512_storeu_si512(dst + i, _mm512_min_epi16(a, _mm512_adds_epi16(constant, b)));
    }
    for (; i < count; ++i)
        dst[i] = std::min(dst[i], add_saturated(cost, row[i]));
}

#endif

/// Non-overlapping buffers let the vectorizer skip its runtime alias checks.
void add_into(Cost* __restrict dst, const Cost* __restrict src, size_t count) {
#if PBQP_X86_DISPATCH
    const Simd level = simd_level();
    if (count >= 32 && level == Simd::avx512) {
        add_into_avx512(dst, src, count);
        return;
    }
    if (count >= 16 && level != Simd::scalar) {
        add_into_avx2(dst, src, count);
        return;
    }
#elif defined(__ARM_NEON)
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const int16x8_t a = vld1q_s16(dst + i);
        const int16x8_t b = vld1q_s16(src + i);
        vst1q_s16(dst + i, vqaddq_s16(a, b));
    }
    for (; i < count; ++i)
        dst[i] = add_saturated(dst[i], src[i]);
    return;
#endif
    for (size_t i = 0; i < count; ++i)
        dst[i] = add_saturated(dst[i], src[i]);
}

void fold_row(Cost* __restrict dst, const Cost* __restrict row, Cost cost, size_t count) {
#if PBQP_X86_DISPATCH
    const Simd level = simd_level();
    if (count >= 32 && level == Simd::avx512) {
        fold_row_avx512(dst, row, cost, count);
        return;
    }
    if (count >= 16 && level != Simd::scalar) {
        fold_row_avx2(dst, row, cost, count);
        return;
    }
#elif defined(__ARM_NEON)
    const int16x8_t constant = vdupq_n_s16(cost);
    size_t i                 = 0;
    for (; i + 8 <= count; i += 8) {
        const int16x8_t a = vld1q_s16(dst + i);
        const int16x8_t b = vld1q_s16(row + i);
        vst1q_s16(dst + i, vminq_s16(a, vqaddq_s16(constant, b)));
    }
    for (; i < count; ++i)
        dst[i] = std::min(dst[i], add_saturated(cost, row[i]));
    return;
#endif
    for (size_t i = 0; i < count; ++i)
        dst[i] = std::min(dst[i], add_saturated(cost, row[i]));
}

/// Snapshot of one edge as seen from the node that is being reduced. The
/// matrix is oriented (reduced option, neighbour option), so it can be
/// evaluated with the neighbour's final choice during back-substitution.
struct TrimmedEdge {
    NodeId neighbour         = 0;
    size_t neighbour_options = 0;
    Matrix matrix;
};

/// Everything needed to restore one node and to choose its option once the
/// choices of its former neighbours are known.
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

/// Copy the matrix of edge `e` into (from option, neighbour option) order.
TrimmedEdge trim_edge(const Graph& graph, EdgeId e, NodeId from) {
    const Graph::Edge& edge = graph.edge(e);
    const bool is_src       = edge.src == from;
    const NodeId other      = is_src ? edge.dst : edge.src;
    const size_t rows       = graph.option_count(from);
    const size_t cols       = graph.option_count(other);

    return TrimmedEdge{
        .neighbour         = other,
        .neighbour_options = cols,
        .matrix            = is_src ? edge.costs : transpose(edge.costs, cols, rows),
    };
}

/// Number of distinct neighbours of a node.
size_t degree(const Graph& graph, NodeId v) { return graph.node(v).edges.size(); }

/// Briggs-style RN heuristic: spread of the cost vector per incident edge.
/// Small values mean "cheap to decide late".
double rn_score(const Graph& graph, NodeId v) {
    const auto [low, high] = std::ranges::minmax(graph.node(v).costs);
    const auto incident    = static_cast<double>(std::max<size_t>(1, degree(graph, v)));
    return static_cast<double>(high - low) / incident;
}

/// Pick the next node to reduce: lowest degree first, then lowest RN score.
std::optional<NodeId> select_node(const Graph& graph) {
    std::optional<NodeId> best;
    size_t best_degree = 0;
    double best_score  = 0.0;

    for (const NodeId v : graph.nodes()) {
        const size_t current_degree = degree(graph, v);
        const double current_score  = rn_score(graph, v);
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

Reduction snapshot(const Graph& graph, NodeId v) {
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
    const TrimmedEdge edge = trim_edge(graph, graph.node(v).edges.front(), v);
    const Vector costs     = graph.node(v).costs;
    const size_t x_dim     = costs.size();
    const size_t y_dim     = edge.neighbour_options;

    Vector& neighbour_costs = graph.node(edge.neighbour).costs;
    Vector best_per_option(y_dim, kInfinity);
    for (size_t x = 0; x < x_dim; ++x)
        fold_row(best_per_option.data(), edge.matrix.data() + x * y_dim, costs[x], y_dim);
    add_into(neighbour_costs.data(), best_per_option.data(), y_dim);
}

/// Add `matrix` (oriented (u, v)) to the edge between u and v, creating it if
/// necessary. RII can create parallel edges, and parallel edges are equivalent
/// to their sum.
void add_or_merge_edge(Graph& graph, NodeId u, NodeId v, const Matrix& matrix) {
    const size_t u_dim = graph.option_count(u);
    const size_t v_dim = graph.option_count(v);
    const size_t count = u_dim * v_dim;

    if (const auto existing = graph.find_edge(u, v)) {
        Graph::Edge& edge = graph.edge(*existing);
        if (edge.src == u) {
            add_into(edge.costs.data(), matrix.data(), count);
        } else {
            const Matrix aligned = transpose(matrix, v_dim, u_dim);
            add_into(edge.costs.data(), aligned.data(), count);
        }
        return;
    }
    graph.link(u, v, matrix);
}

/// R2: contract a degree-2 node `v` with neighbours u and w.
///
///     M'_uw(i, j) = min_x ( c_v(x) + M_uv(x, i) + M_vw(x, j) )
void contract_node(Graph& graph, NodeId v) {
    const TrimmedEdge to_u = trim_edge(graph, graph.node(v).edges[0], v);
    const TrimmedEdge to_w = trim_edge(graph, graph.node(v).edges[1], v);
    const Vector costs     = graph.node(v).costs;

    const size_t u_dim = to_u.neighbour_options;
    const size_t w_dim = to_w.neighbour_options;
    const size_t x_dim = costs.size();

    Matrix merged(u_dim * w_dim, kInfinity);
    for (size_t x = 0; x < x_dim; ++x) {
        const Cost* const w_row = to_w.matrix.data() + x * w_dim;
        for (size_t i = 0; i < u_dim; ++i)
            fold_row(merged.data() + i * w_dim, w_row, add_saturated(costs[x], to_u.matrix[x * u_dim + i]), w_dim);
    }
    add_or_merge_edge(graph, to_u.neighbour, to_w.neighbour, merged);
}

/// Choose the cheapest option for a restored node, given the choices of the
/// neighbours captured in its reduction. Exact for snapshots of degree <= 2
/// nodes, greedy otherwise.
OptionIndex back_substitute(const Reduction& reduction, const Choices& choices) {
    OptionIndex best = 0;
    Cost best_total  = kInfinity;

    for (OptionIndex option = 0; option < reduction.costs.size(); ++option) {
        Cost total = reduction.costs[option];
        for (const TrimmedEdge& edge : reduction.edges)
            total = add_saturated(total, edge.matrix[option * edge.neighbour_options + choices[edge.neighbour]]);
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

    for (const NodeId v : graph.nodes()) {
        if (choices.size() <= v) throw std::invalid_argument("evaluate: choices is too short");
        const auto& costs = graph.node(v).costs;
        if (choices[v] >= costs.size()) throw std::invalid_argument("evaluate: option index out of range");
        total += static_cast<CostSum>(costs[choices[v]]);
    }
    for (const EdgeId e : graph.edges()) {
        const Graph::Edge& edge = graph.edge(e);
        total += static_cast<CostSum>(
            edge.costs.at(choices[edge.src] * graph.option_count(edge.dst) + choices[edge.dst]));
    }
    return total;
}

Solution solve(const Graph& graph) {
    Graph working = graph;
    ankerl::svector<Reduction, 8> stack;
    stack.reserve(graph.node_count());

    while (const auto v = select_node(working)) {
        const size_t d      = degree(working, *v);
        Reduction reduction = snapshot(working, *v);

        if (d == 1)
            fold_into_neighbour(working, *v);
        else if (d == 2)
            contract_node(working, *v);

        working.remove_node(*v);
        stack.push_back(std::move(reduction));
    }

    size_t slots = 0;
    for (const NodeId v : graph.nodes())
        slots = std::max(slots, v + 1);

    Choices choices(slots, 0);
    for (auto it = stack.rbegin(); it != stack.rend(); ++it)
        choices[it->node] = back_substitute(*it, choices);

    const CostSum cost = evaluate(graph, choices);
    return Solution{.choices = std::move(choices), .cost = cost};
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

Enumeration make_enumeration(const Graph& graph, size_t max_combinations) {
    Enumeration enumeration;
    for (const NodeId v : graph.nodes()) {
        enumeration.nodes.push_back(v);
        enumeration.options.push_back(graph.option_count(v));
    }
    if (enumeration.nodes.empty()) return enumeration;

    enumeration.slots = std::ranges::max(enumeration.nodes) + 1;
    enumeration.place.resize(enumeration.nodes.size());

    size_t combinations = 1;
    for (const size_t options : enumeration.options) {
        if (combinations > max_combinations / options) throw std::runtime_error("solve_exhaustive: instance too large");
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

/// Scan combinations [begin, end); the smallest index wins ties so that the
/// result does not depend on how the space is partitioned.
BestAssignment search_range(const Graph& graph, const Enumeration& enumeration, size_t begin, size_t end) {
    Choices choices(enumeration.slots, 0);
    for (size_t i = 0; i < enumeration.nodes.size(); ++i)
        choices[enumeration.nodes[i]] = (begin / enumeration.place[i]) % enumeration.options[i];

    BestAssignment best{
        .solution = Solution{.choices = choices, .cost = kMaxCost},
        .index    = end
    };
    for (size_t index = begin; index < end; ++index) {
        const CostSum cost = evaluate(graph, choices);
        if (cost < best.solution.cost) {
            best.solution = Solution{.choices = choices, .cost = cost};
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

Solution solve_exhaustive(const Graph& graph, size_t max_combinations, unsigned thread_count) {
    const Enumeration enumeration = make_enumeration(graph, max_combinations);
    if (enumeration.nodes.empty()) return Solution{};

    unsigned requested = thread_count;
    if (requested == 0) requested = std::thread::hardware_concurrency();
    if (requested == 0) requested = 1;
    const size_t workers = std::min<size_t>(requested, enumeration.combinations);

    if (workers <= 1) return search_range(graph, enumeration, 0, enumeration.combinations).solution;

    ankerl::svector<BestAssignment, 8> partials(workers);
    ankerl::svector<std::thread, 8> threads;
    threads.reserve(workers);

    const size_t base  = enumeration.combinations / workers;
    const size_t extra = enumeration.combinations % workers;

    try {
        size_t begin = 0;
        for (size_t i = 0; i < workers; ++i) {
            const size_t end = begin + base + (i < extra ? 1 : 0);
            threads.push_back(std::thread([&graph, &enumeration, &partials, i, begin, end] {
                partials[i] = search_range(graph, enumeration, begin, end);
            }));
            begin = end;
        }
    } catch (...) {
        for (auto& thread : threads)
            thread.join();
        throw;
    }
    for (auto& thread : threads)
        thread.join();

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
