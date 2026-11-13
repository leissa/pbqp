#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <array>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <thread>
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

void single_node_picks_cheapest_option() {
    pbqp::Graph graph;
    const pbqp::NodeId v = graph.add_node({5, 1, 3});

    const pbqp::Solution solution = pbqp::solve(graph);
    CHECK(solution.choices[v] == 1);
    CHECK(solution.cost == 1);
}

void independent_nodes_are_independent() {
    pbqp::Graph graph;
    const pbqp::NodeId a = graph.add_node({4, 2});
    const pbqp::NodeId b = graph.add_node({7, 8, 6});

    const pbqp::Solution solution = pbqp::solve(graph);
    CHECK(solution.choices[a] == 1);
    CHECK(solution.choices[b] == 2);
    CHECK(solution.cost == 8);
}

void edge_choice_trades_off_node_costs() {
    // The cheap option of `a` is expensive once the edge is paid for.
    pbqp::Graph graph;
    const pbqp::NodeId a = graph.add_node({0, 1});
    const pbqp::NodeId b = graph.add_node({1, 0});
    graph.link(a, b,
               {
                   10,
                   10, // a=0
                   0,
                   0, // a=1
               });

    const pbqp::Solution solution = pbqp::solve(graph);
    CHECK(solution.choices[a] == 1);
    CHECK(solution.choices[b] == 1);
    CHECK(solution.cost == 1);
}

void triangle_with_conflicting_colours() {
    // Two colours, an odd cycle, and a penalty for monochromatic edges: one
    // edge has to be paid for no matter what.
    pbqp::Graph graph;
    ankerl::svector<pbqp::NodeId, 4> nodes;
    for (int i = 0; i < 3; ++i)
        nodes.push_back(graph.add_node({0, 0}));

    const ankerl::svector<pbqp::Cost, 4> same_colour = {
        5,
        0,
        0,
        5,
    };
    for (size_t i = 0; i < nodes.size(); ++i)
        graph.link(nodes[i], nodes[(i + 1) % nodes.size()], same_colour);

    const pbqp::Solution solution = pbqp::solve(graph);
    CHECK(solution.cost == 5);
    CHECK(solution.cost == pbqp::evaluate(graph, solution.choices));
}

/// Random tree: RI/RII are exact there, so the heuristic must be optimal.
void heuristic_is_optimal_on_random_trees() {
    std::mt19937_64 random{7};
    std::uniform_int_distribution<int> cost{0, 10};
    const auto next_cost = [&] { return static_cast<pbqp::Cost>(cost(random)); };

    for (int trial = 0; trial < 40; ++trial) {
        const size_t node_count = 2 + static_cast<size_t>(trial % 7);
        const size_t options    = 2 + static_cast<size_t>(trial % 3);

        pbqp::Graph graph;
        ankerl::svector<pbqp::Cost, 4> values(options);
        for (size_t i = 0; i < node_count; ++i) {
            std::ranges::generate(values, next_cost);
            graph.add_node(values);
        }

        ankerl::svector<pbqp::Cost, 4> matrix(options * options);
        for (size_t i = 1; i < node_count; ++i) {
            std::ranges::generate(matrix, next_cost);
            std::uniform_int_distribution<size_t> parent{0, i - 1};
            graph.link(parent(random), i, matrix);
        }

        const pbqp::Solution heuristic = pbqp::solve(graph);
        const pbqp::Solution optimal   = pbqp::solve_exhaustive(graph);
        CHECK(heuristic.cost == optimal.cost);
        CHECK(heuristic.cost == pbqp::evaluate(graph, heuristic.choices));
    }
}

/// On general graphs the heuristic may lose, but it must return a valid
/// assignment whose reported cost matches a fresh evaluation.
void heuristic_result_is_always_valid() {
    std::mt19937_64 random{11};
    std::uniform_int_distribution<int> cost{0, 10};
    const auto next_cost = [&] { return static_cast<pbqp::Cost>(cost(random)); };

    for (int trial = 0; trial < 40; ++trial) {
        const size_t node_count = 3 + static_cast<size_t>(trial % 5);
        const size_t options    = 2 + static_cast<size_t>(trial % 2);

        pbqp::Graph graph;
        ankerl::svector<pbqp::Cost, 4> values(options);
        for (size_t i = 0; i < node_count; ++i) {
            std::ranges::generate(values, next_cost);
            graph.add_node(values);
        }

        ankerl::svector<pbqp::Cost, 4> matrix(options * options);
        for (size_t u = 0; u < node_count; ++u) {
            for (size_t v = u + 1; v < node_count; ++v) {
                if (cost(random) < 5) continue; // roughly half the pairs
                std::ranges::generate(matrix, next_cost);
                graph.link(u, v, matrix);
            }
        }

        const pbqp::Solution heuristic = pbqp::solve(graph);
        const pbqp::Solution optimal   = pbqp::solve_exhaustive(graph);

        CHECK(heuristic.choices.size() == node_count);
        CHECK(heuristic.cost == pbqp::evaluate(graph, heuristic.choices));
        CHECK(heuristic.cost >= optimal.cost);
    }
}

void negative_costs_reward_coalescing() {
    pbqp::Graph graph;
    const pbqp::NodeId a = graph.add_node({0, 0});
    const pbqp::NodeId b = graph.add_node({0, 0});
    graph.link(a, b,
               {
                   -5,
                   0,
                   0,
                   -5,
               });

    const pbqp::Solution solution = pbqp::solve(graph);
    CHECK(solution.cost == -5);
    CHECK(solution.choices[a] == solution.choices[b]);
}

void infinity_forbids_same_option() {
    constexpr auto infinity = std::numeric_limits<pbqp::Cost>::max();

    pbqp::Graph graph;
    const pbqp::NodeId a = graph.add_node({0, 0});
    const pbqp::NodeId b = graph.add_node({0, 0});
    graph.link(a, b,
               {
                   infinity,
                   0,
                   0,
                   infinity,
               });

    const pbqp::Solution solution = pbqp::solve(graph);
    CHECK(solution.choices[a] != solution.choices[b]);
    CHECK(solution.cost == 0);
    CHECK(solution.cost == pbqp::evaluate(graph, solution.choices));
}

void saturated_sums_do_not_wrap() {
    constexpr auto infinity = std::numeric_limits<pbqp::Cost>::max();

    // Three nodes, two colours: one monochromatic edge is unavoidable. The
    // reduction sums several infinities, which must saturate instead of
    // wrapping into a negative cost.
    pbqp::Graph graph;
    ankerl::svector<pbqp::NodeId, 4> nodes;
    for (int i = 0; i < 3; ++i)
        nodes.push_back(graph.add_node({0, 0}));

    const ankerl::svector<pbqp::Cost, 4> interference = {
        infinity,
        0,
        0,
        infinity,
    };
    for (size_t i = 0; i < nodes.size(); ++i)
        graph.link(nodes[i], nodes[(i + 1) % nodes.size()], interference);

    const pbqp::Solution heuristic = pbqp::solve(graph);
    const pbqp::Solution optimal   = pbqp::solve_exhaustive(graph);
    CHECK(heuristic.cost == optimal.cost);
    CHECK(heuristic.cost == infinity);
    CHECK(heuristic.cost == pbqp::evaluate(graph, heuristic.choices));
}

void evaluate_rejects_bad_assignments() {
    pbqp::Graph graph;
    static_cast<void>(graph.add_node({1, 2}));

    CHECK_THROWS(pbqp::evaluate(graph, pbqp::Choices{}));
    CHECK_THROWS(pbqp::evaluate(graph, pbqp::Choices{2}));
    CHECK(pbqp::evaluate(graph, pbqp::Choices{1}) == 2);
}

void graph_rejects_invalid_input() {
    pbqp::Graph graph;
    const pbqp::NodeId a = graph.add_node({1, 2});
    const pbqp::NodeId b = graph.add_node({1, 2});

    CHECK_THROWS(graph.add_node({}));
    CHECK_THROWS(graph.link(a, a, {0, 0, 0, 0})); // self edge
    CHECK_THROWS(graph.link(a, b, {0, 0}));       // wrong size
    graph.link(a, b, {0, 0, 0, 0});
    CHECK_THROWS(graph.link(b, a, {0, 0, 0, 0})); // duplicate
    CHECK(graph.find_edge(a, b).has_value());
    CHECK(graph.find_edge(b, a).has_value());
}

void removal_keeps_ids_stable() {
    pbqp::Graph graph;
    for (int i = 0; i < 4; ++i)
        graph.add_node({1, 2});
    graph.link(0, 1, {0, 0, 0, 0});
    graph.link(1, 2, {0, 0, 0, 0});

    CHECK(graph.node_count() == 4);
    CHECK(graph.edge_count() == 2);

    graph.remove_node(1);
    CHECK(graph.has_node(0));
    CHECK(!graph.has_node(1));
    CHECK(graph.has_node(2));
    CHECK(graph.has_node(3));
    CHECK(graph.node_count() == 3);
    CHECK(graph.edge_count() == 0);
    CHECK(!graph.find_edge(0, 2).has_value());

    const pbqp::Solution solution = pbqp::solve(graph);
    CHECK(solution.cost == 3);
}

void parallel_exhaustive_matches_sequential() {
    std::mt19937_64 random{29};
    std::uniform_int_distribution<int> cost{0, 10};
    const auto next_cost = [&] { return static_cast<pbqp::Cost>(cost(random)); };

    pbqp::Graph graph;
    ankerl::svector<pbqp::Cost, 4> values(3);
    for (size_t i = 0; i < 6; ++i) {
        std::ranges::generate(values, next_cost);
        graph.add_node(values);
    }

    ankerl::svector<pbqp::Cost, 4> matrix(9);
    for (size_t u = 0; u < 6; ++u) {
        for (size_t v = u + 1; v < 6; ++v) {
            if (cost(random) < 5) continue;
            std::ranges::generate(matrix, next_cost);
            graph.link(u, v, matrix);
        }
    }

    const pbqp::Solution sequential = pbqp::solve_exhaustive(graph, 1'000'000, 1);
    const pbqp::Solution parallel   = pbqp::solve_exhaustive(graph, 1'000'000, 4);
    CHECK(sequential.choices == parallel.choices);
    CHECK(sequential.cost == parallel.cost);
}

void wide_option_vectors_stay_exact() {
    std::mt19937_64 random{41};
    std::uniform_int_distribution<int> cost{0, 100};
    const auto next_cost = [&] { return static_cast<pbqp::Cost>(cost(random)); };

    for (const size_t options : {size_t{17}, size_t{33}}) {
        ankerl::svector<pbqp::Cost, 4> values(options);
        ankerl::svector<pbqp::Cost, 4> matrix(options * options);

        // A single edge exercises RI with a wide row.
        pbqp::Graph pair;
        std::ranges::generate(values, next_cost);
        const pbqp::NodeId a = pair.add_node(values);
        std::ranges::generate(values, next_cost);
        const pbqp::NodeId b = pair.add_node(values);
        std::ranges::generate(matrix, next_cost);
        pair.link(a, b, matrix);
        CHECK(pbqp::solve(pair).cost == pbqp::solve_exhaustive(pair).cost);

        // A triangle has no low-degree node, so RII contracts a wide matrix.
        pbqp::Graph triangle;
        ankerl::svector<pbqp::NodeId, 4> nodes;
        for (int i = 0; i < 3; ++i) {
            std::ranges::generate(values, next_cost);
            nodes.push_back(triangle.add_node(values));
        }
        for (size_t i = 0; i < nodes.size(); ++i) {
            std::ranges::generate(matrix, next_cost);
            triangle.link(nodes[i], nodes[(i + 1) % nodes.size()], matrix);
        }
        CHECK(pbqp::solve(triangle).cost == pbqp::solve_exhaustive(triangle).cost);
    }
}

void concurrent_solves_share_a_graph() {
    std::mt19937_64 random{31};
    std::uniform_int_distribution<int> cost{0, 10};
    const auto next_cost = [&] { return static_cast<pbqp::Cost>(cost(random)); };

    pbqp::Graph graph;
    ankerl::svector<pbqp::Cost, 4> values(2);
    for (size_t i = 0; i < 7; ++i) {
        std::ranges::generate(values, next_cost);
        graph.add_node(values);
    }

    ankerl::svector<pbqp::Cost, 4> matrix(4);
    for (size_t u = 0; u < 7; ++u) {
        for (size_t v = u + 1; v < 7; ++v) {
            if (cost(random) < 5) continue;
            std::ranges::generate(matrix, next_cost);
            graph.link(u, v, matrix);
        }
    }

    const pbqp::Solution heuristic = pbqp::solve(graph);
    const pbqp::Solution optimal   = pbqp::solve_exhaustive(graph, 1'000'000, 1);

    constexpr size_t thread_count = 4;
    std::array<pbqp::Solution, thread_count> heuristics;
    std::array<pbqp::Solution, thread_count> optima;
    std::array<std::thread, thread_count> workers;

    for (size_t i = 0; i < thread_count; ++i) {
        workers[i] = std::thread([&, i] {
            heuristics[i] = pbqp::solve(graph);
            optima[i]     = pbqp::solve_exhaustive(graph, 1'000'000, 2);
        });
    }
    for (auto& worker : workers)
        worker.join();

    for (size_t i = 0; i < thread_count; ++i) {
        CHECK(heuristics[i].cost == heuristic.cost);
        CHECK(optima[i].choices == optimal.choices);
    }
}

} // namespace

int main() {
    single_node_picks_cheapest_option();
    independent_nodes_are_independent();
    edge_choice_trades_off_node_costs();
    triangle_with_conflicting_colours();
    heuristic_is_optimal_on_random_trees();
    heuristic_result_is_always_valid();
    negative_costs_reward_coalescing();
    infinity_forbids_same_option();
    saturated_sums_do_not_wrap();
    evaluate_rejects_bad_assignments();
    graph_rejects_invalid_input();
    removal_keeps_ids_stable();
    parallel_exhaustive_matches_sequential();
    wide_option_vectors_stay_exact();
    concurrent_solves_share_a_graph();

    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "all checks passed\n";
    return 0;
}
