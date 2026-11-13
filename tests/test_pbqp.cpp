#include <cmath>
#include <cstddef>

#include <algorithm>
#include <exception>
#include <iostream>
#include <random>
#include <utility>

#include <ankerl/svector.h>

#include "pbqp/graph.hpp"
#include "pbqp/solver.hpp"

namespace {

int failures = 0;

#define CHECK(condition)                                                                    \
    do {                                                                                    \
        if (!(condition)) {                                                                 \
            std::cerr << __FILE__ << ':' << __LINE__ << ": CHECK failed: " #condition "\n"; \
            ++failures;                                                                     \
        }                                                                                   \
    } while (false)

#define CHECK_THROWS(expression)                                                                          \
    do {                                                                                                  \
        bool threw = false;                                                                               \
        try {                                                                                             \
            static_cast<void>(expression);                                                                \
        } catch (const std::exception&) { threw = true; }                                                 \
        if (!threw) {                                                                                     \
            std::cerr << __FILE__ << ':' << __LINE__ << ": expected an exception from " #expression "\n"; \
            ++failures;                                                                                   \
        }                                                                                                 \
    } while (false)

auto nearly_equal(double lhs, double rhs, double tolerance = 1e-9) -> bool {
    return std::abs(lhs - rhs) <= tolerance * std::max({1.0, std::abs(lhs), std::abs(rhs)});
}

void single_node_picks_cheapest_option() {
    pbqp::Graph graph;
    const pbqp::NodeId v = graph.add_node({5.0, 1.0, 3.0});

    const pbqp::Solution solution = pbqp::solve(graph);
    CHECK(solution.choices[v] == 1);
    CHECK(nearly_equal(solution.cost, 1.0));
}

void independent_nodes_are_independent() {
    pbqp::Graph graph;
    const pbqp::NodeId a = graph.add_node({4.0, 2.0});
    const pbqp::NodeId b = graph.add_node({7.0, 8.0, 6.0});

    const pbqp::Solution solution = pbqp::solve(graph);
    CHECK(solution.choices[a] == 1);
    CHECK(solution.choices[b] == 2);
    CHECK(nearly_equal(solution.cost, 8.0));
}

void edge_choice_trades_off_node_costs() {
    // The cheap option of `a` is expensive once the edge is paid for.
    pbqp::Graph graph;
    const pbqp::NodeId a = graph.add_node({0.0, 1.0});
    const pbqp::NodeId b = graph.add_node({1.0, 0.0});
    graph.add_edge(a, b,
                   {
                       10.0,
                       10.0, // a=0
                       0.0,
                       0.0, // a=1
                   });

    const pbqp::Solution solution = pbqp::solve(graph);
    CHECK(solution.choices[a] == 1);
    CHECK(solution.choices[b] == 1);
    CHECK(nearly_equal(solution.cost, 1.0));
}

void triangle_with_conflicting_colours() {
    // Two colours, an odd cycle, and a penalty for monochromatic edges: one
    // edge has to be paid for no matter what.
    pbqp::Graph graph;
    ankerl::svector<pbqp::NodeId, 4> nodes;
    for (int i = 0; i < 3; ++i)
        nodes.push_back(graph.add_node({0.0, 0.0}));

    const ankerl::svector<double, 4> same_colour = {
        5.0,
        0.0,
        0.0,
        5.0,
    };
    for (std::size_t i = 0; i < nodes.size(); ++i)
        graph.add_edge(nodes[i], nodes[(i + 1) % nodes.size()], same_colour);

    const pbqp::Solution solution = pbqp::solve(graph);
    CHECK(nearly_equal(solution.cost, 5.0));
    CHECK(nearly_equal(solution.cost, pbqp::evaluate(graph, solution.choices)));
}

/// Random tree: RI/RII are exact there, so the heuristic must be optimal.
void heuristic_is_optimal_on_random_trees() {
    std::mt19937_64 random{7};
    std::uniform_real_distribution<double> cost{0.0, 10.0};

    for (int trial = 0; trial < 40; ++trial) {
        const std::size_t node_count = 2 + static_cast<std::size_t>(trial % 7);
        const std::size_t options    = 2 + static_cast<std::size_t>(trial % 3);

        pbqp::Graph graph;
        ankerl::svector<double, 4> values(options);
        for (std::size_t i = 0; i < node_count; ++i) {
            std::ranges::generate(values, [&] { return cost(random); });
            graph.add_node(values);
        }

        ankerl::svector<double, 4> matrix(options * options);
        for (std::size_t i = 1; i < node_count; ++i) {
            std::ranges::generate(matrix, [&] { return cost(random); });
            std::uniform_int_distribution<std::size_t> parent{0, i - 1};
            graph.add_edge(parent(random), i, matrix);
        }

        const pbqp::Solution heuristic = pbqp::solve(graph);
        const pbqp::Solution optimal   = pbqp::solve_exhaustive(graph);
        CHECK(nearly_equal(heuristic.cost, optimal.cost));
        CHECK(nearly_equal(heuristic.cost, pbqp::evaluate(graph, heuristic.choices)));
    }
}

/// On general graphs the heuristic may lose, but it must return a valid
/// assignment whose reported cost matches a fresh evaluation.
void heuristic_result_is_always_valid() {
    std::mt19937_64 random{11};
    std::uniform_real_distribution<double> cost{0.0, 10.0};

    for (int trial = 0; trial < 40; ++trial) {
        const std::size_t node_count = 3 + static_cast<std::size_t>(trial % 5);
        const std::size_t options    = 2 + static_cast<std::size_t>(trial % 2);

        pbqp::Graph graph;
        ankerl::svector<double, 4> values(options);
        for (std::size_t i = 0; i < node_count; ++i) {
            std::ranges::generate(values, [&] { return cost(random); });
            graph.add_node(values);
        }

        ankerl::svector<double, 4> matrix(options * options);
        for (std::size_t u = 0; u < node_count; ++u) {
            for (std::size_t v = u + 1; v < node_count; ++v) {
                if (cost(random) < 5.0) continue; // roughly half the pairs
                std::ranges::generate(matrix, [&] { return cost(random); });
                graph.add_edge(u, v, matrix);
            }
        }

        const pbqp::Solution heuristic = pbqp::solve(graph);
        const pbqp::Solution optimal   = pbqp::solve_exhaustive(graph);

        CHECK(heuristic.choices.size() == node_count);
        CHECK(nearly_equal(heuristic.cost, pbqp::evaluate(graph, heuristic.choices)));
        CHECK(heuristic.cost >= optimal.cost - 1e-9);
    }
}

void evaluate_rejects_bad_assignments() {
    pbqp::Graph graph;
    const pbqp::NodeId v = graph.add_node({1.0, 2.0});

    CHECK_THROWS(pbqp::evaluate(graph, pbqp::Choices{}));
    CHECK_THROWS(pbqp::evaluate(graph, pbqp::Choices{2}));
    CHECK(nearly_equal(pbqp::evaluate(graph, pbqp::Choices{1}), 2.0));

    static_cast<void>(v);
}

void graph_rejects_invalid_input() {
    pbqp::Graph graph;
    const pbqp::NodeId a = graph.add_node({1.0, 2.0});
    const pbqp::NodeId b = graph.add_node({1.0, 2.0});

    CHECK_THROWS(graph.add_node({}));
    CHECK_THROWS(graph.add_edge(a, a, {0.0, 0.0, 0.0, 0.0})); // self edge
    CHECK_THROWS(graph.add_edge(a, b, {0.0, 0.0}));           // wrong size
    graph.add_edge(a, b, {0.0, 0.0, 0.0, 0.0});
    CHECK_THROWS(graph.add_edge(b, a, {0.0, 0.0, 0.0, 0.0})); // duplicate
    CHECK(graph.find_edge(a, b).has_value());
    CHECK(graph.find_edge(b, a).has_value());
}

void removal_keeps_ids_stable() {
    pbqp::Graph graph;
    for (int i = 0; i < 4; ++i)
        graph.add_node({1.0, 2.0});
    graph.add_edge(0, 1, {0.0, 0.0, 0.0, 0.0});
    graph.add_edge(1, 2, {0.0, 0.0, 0.0, 0.0});

    graph.remove_node(1);
    CHECK(graph.has_node(0));
    CHECK(!graph.has_node(1));
    CHECK(graph.has_node(2));
    CHECK(graph.has_node(3));
    CHECK(graph.node_count() == 3);
    CHECK(graph.edge_count() == 0);
    CHECK(!graph.find_edge(0, 2).has_value());

    const pbqp::Solution solution = pbqp::solve(graph);
    CHECK(nearly_equal(solution.cost, 3.0));
}

} // namespace

int main() {
    single_node_picks_cheapest_option();
    independent_nodes_are_independent();
    edge_choice_trades_off_node_costs();
    triangle_with_conflicting_colours();
    heuristic_is_optimal_on_random_trees();
    heuristic_result_is_always_valid();
    evaluate_rejects_bad_assignments();
    graph_rejects_invalid_input();
    removal_keeps_ids_stable();

    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "all checks passed\n";
    return 0;
}
