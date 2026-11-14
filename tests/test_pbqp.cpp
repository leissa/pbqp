#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <array>
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

void single_node_picks_cheapest_option() {
    pbqp::Graph graph;
    const pbqp::NodeId n = graph.add_node({5, 1, 3});

    const pbqp::Solution solution = pbqp::solve(graph);
    CHECK(solution.choices[n] == 1);
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
        nodes.emplace_back(graph.add_node({0, 0}));

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
void heuristic_is_optimal_on_random_trees(int min_cost, int max_cost, int forbidden_percent) {
    std::mt19937_64 random{7};
    std::uniform_int_distribution<int> cost{min_cost, max_cost};
    std::uniform_int_distribution<int> percent{0, 99};
    std::uniform_int_distribution<size_t> option_count{1, 4};
    const auto next_cost
        = [&] { return percent(random) < forbidden_percent ? pbqp::kInfinity : static_cast<pbqp::Cost>(cost(random)); };

    for (int trial = 0; trial < 200; ++trial) {
        const size_t node_count = 2 + static_cast<size_t>(trial % 7);

        pbqp::Graph graph;
        for (size_t i = 0; i < node_count; ++i) {
            ankerl::svector<pbqp::Cost, 4> values(option_count(random));
            std::ranges::generate(values, next_cost);
            graph.add_node(values);
        }

        for (pbqp::NodeId n = 1; n < node_count; ++n) {
            const pbqp::NodeId m = std::uniform_int_distribution<size_t>{0, n - 1}(random);
            ankerl::svector<pbqp::Cost, 4> matrix(graph.option_count(m) * graph.option_count(n));
            std::ranges::generate(matrix, next_cost);
            if (cost(random) % 2 == 0)
                graph.link(m, n, matrix);
            else
                graph.link(n, m, matrix);
        }

        const pbqp::Solution heuristic = pbqp::solve(graph);
        const pbqp::Solution optimal   = *pbqp::solve_exhaustive(graph);
        CHECK(heuristic.cost == optimal.cost);
        CHECK(heuristic.cost == pbqp::evaluate(graph, heuristic.choices));
    }
}

void long_path_with_large_costs_stays_exact() {
    pbqp::Graph graph;
    pbqp::NodeId previous = graph.add_node({4000, 4000});
    for (int i = 0; i < 11; ++i) {
        const pbqp::NodeId next = graph.add_node({4000, 4000});
        graph.link(previous, next, {0, 50, 50, 0});
        previous = next;
    }
    const pbqp::NodeId last = graph.add_node({100, 0});
    graph.link(previous, last, {0, 50, 50, 0});

    CHECK(pbqp::solve(graph).cost == pbqp::solve_exhaustive(graph)->cost);
}

void rn_charges_its_edges_to_the_neighbours() {
    // Dense instance found by search: without charging the RN rows to the neighbours the result is 51.
    pbqp::Graph graph;
    graph.add_node({8, 1, 9});
    graph.add_node({0, 4, 1});
    graph.add_node({0, 6, 6});
    graph.add_node({0, 9, 2});
    graph.add_node({7, 0, 4});
    graph.link(0, 1, {8, 3, 9, 9, 4, 8, 6, 0, 3});
    graph.link(0, 2, {9, 1, 2, 9, 5, 0, 4, 3, 9});
    graph.link(0, 3, {4, 3, 4, 7, 7, 6, 0, 7, 9});
    graph.link(0, 4, {5, 6, 0, 3, 3, 8, 8, 4, 6});
    graph.link(1, 2, {9, 4, 6, 9, 1, 4, 4, 1, 6});
    graph.link(1, 3, {6, 2, 8, 2, 1, 8, 7, 2, 7});
    graph.link(1, 4, {2, 4, 0, 1, 1, 4, 0, 1, 3});
    graph.link(2, 3, {4, 1, 4, 1, 8, 5, 7, 5, 5});
    graph.link(2, 4, {9, 5, 4, 4, 5, 2, 2, 1, 0});
    graph.link(3, 4, {6, 9, 6, 5, 1, 9, 2, 6, 9});

    CHECK(pbqp::solve(graph).cost == 45);
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
        for (size_t m = 0; m < node_count; ++m) {
            for (size_t n = m + 1; n < node_count; ++n) {
                if (cost(random) < 5) continue; // roughly half the pairs
                std::ranges::generate(matrix, next_cost);
                graph.link(m, n, matrix);
            }
        }

        const pbqp::Solution heuristic = pbqp::solve(graph);
        const pbqp::Solution optimal   = *pbqp::solve_exhaustive(graph);

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

void infinite_sums_do_not_wrap() {
    constexpr auto infinity = std::numeric_limits<pbqp::Cost>::max();

    // Three nodes and two colours make one monochromatic edge unavoidable, so the reduction sums several infinities.
    pbqp::Graph graph;
    ankerl::svector<pbqp::NodeId, 4> nodes;
    for (int i = 0; i < 3; ++i)
        nodes.emplace_back(graph.add_node({0, 0}));

    const ankerl::svector<pbqp::Cost, 4> interference = {
        infinity,
        0,
        0,
        infinity,
    };
    for (size_t i = 0; i < nodes.size(); ++i)
        graph.link(nodes[i], nodes[(i + 1) % nodes.size()], interference);

    const pbqp::Solution heuristic = pbqp::solve(graph);
    const pbqp::Solution optimal   = *pbqp::solve_exhaustive(graph);
    CHECK(heuristic.cost == optimal.cost);
    CHECK(heuristic.cost == pbqp::kInfiniteSum);
    CHECK(heuristic.cost == pbqp::evaluate(graph, heuristic.choices));
}

/// Complete graph whose nodes all keep one option after pruning, except for `free_nodes`, which then form a tree.
pbqp::Graph pruned_complete_graph(std::mt19937_64& random, size_t node_count, size_t free_nodes) {
    std::uniform_int_distribution<int> cost{0, 10};
    std::uniform_int_distribution<size_t> option_count{2, 4};
    const auto next_cost = [&] { return static_cast<pbqp::Cost>(cost(random)); };

    pbqp::Graph graph;
    for (size_t i = 0; i < node_count; ++i) {
        pbqp::Costs values(option_count(random));
        std::ranges::generate(values, next_cost);
        if (i >= free_nodes) {
            const size_t kept = std::uniform_int_distribution<size_t>{0, values.size() - 1}(random);
            for (size_t x = 0; x < values.size(); ++x)
                if (x != kept) values[x] = pbqp::kInfinity;
        }
        graph.add_node(values);
    }
    for (pbqp::NodeId m = 0; m < node_count; ++m) {
        for (pbqp::NodeId n = m + 1; n < node_count; ++n) {
            pbqp::Costs matrix(graph.option_count(m) * graph.option_count(n));
            std::ranges::generate(matrix, next_cost);
            graph.link(m, n, matrix);
        }
    }
    return graph;
}

void pruning_makes_dense_graphs_exact() {
    std::mt19937_64 random{53};
    for (int trial = 0; trial < 100; ++trial) {
        const pbqp::Graph graph        = pruned_complete_graph(random, 6, 2);
        const pbqp::Solution heuristic = pbqp::solve(graph);
        CHECK(heuristic.cost == pbqp::solve_exhaustive(graph)->cost);
        CHECK(heuristic.cost == pbqp::evaluate(graph, heuristic.choices));
    }
}

void heuristic_handles_forbidden_options_on_dense_graphs() {
    std::mt19937_64 random{59};
    std::uniform_int_distribution<int> cost{0, 10};
    std::uniform_int_distribution<int> percent{0, 99};
    std::uniform_int_distribution<size_t> option_count{1, 4};
    const auto next_cost
        = [&] { return percent(random) < 25 ? pbqp::kInfinity : static_cast<pbqp::Cost>(cost(random)); };

    for (int trial = 0; trial < 200; ++trial) {
        const size_t node_count = 4 + static_cast<size_t>(trial % 4);
        pbqp::Graph graph;
        for (size_t i = 0; i < node_count; ++i) {
            pbqp::Costs values(option_count(random));
            std::ranges::generate(values, next_cost);
            graph.add_node(values);
        }
        for (pbqp::NodeId m = 0; m < node_count; ++m) {
            for (pbqp::NodeId n = m + 1; n < node_count; ++n) {
                if (percent(random) < 30) continue;
                pbqp::Costs matrix(graph.option_count(m) * graph.option_count(n));
                std::ranges::generate(matrix, next_cost);
                graph.link(m, n, matrix);
            }
        }

        const pbqp::Solution heuristic = pbqp::solve(graph);
        CHECK(heuristic.cost == pbqp::evaluate(graph, heuristic.choices));
        CHECK(heuristic.cost >= pbqp::solve_exhaustive(graph)->cost);
    }
}

struct IntervalInstance {
    pbqp::Graph graph;
    ankerl::svector<size_t, 8> peo_position;
};

/// Interval graphs are chordal, so k = maximum overlap registers always suffice; affinities add coalescing costs.
IntervalInstance random_interval_instance(std::mt19937_64& random, size_t node_count) {
    std::uniform_int_distribution<int> start{0, 40};
    std::uniform_int_distribution<int> length{1, 12};
    std::uniform_int_distribution<int> weight{1, 100};
    std::uniform_int_distribution<int> percent{0, 99};

    ankerl::svector<std::pair<int, int>, 8> intervals;
    for (size_t i = 0; i < node_count; ++i) {
        const int begin = start(random);
        intervals.emplace_back(begin, begin + length(random));
    }
    const auto interferes = [&](size_t a, size_t b) {
        return intervals[a].first < intervals[b].second && intervals[b].first < intervals[a].second;
    };

    size_t registers = 1;
    for (const auto& [begin, end] : intervals) {
        const auto live = std::ranges::count_if(
            intervals, [&](const auto& other) { return other.first <= begin && begin < other.second; });
        registers = std::max(registers, static_cast<size_t>(live));
    }

    IntervalInstance instance;
    for (size_t i = 0; i < node_count; ++i)
        instance.graph.add_node(pbqp::Costs(registers, 0));
    for (pbqp::NodeId m = 0; m < node_count; ++m) {
        for (pbqp::NodeId n = m + 1; n < node_count; ++n) {
            const bool affinity = percent(random) < 15;
            if (!interferes(m, n) && !affinity) continue;
            const pbqp::Cost copy = affinity ? static_cast<pbqp::Cost>(weight(random)) : 0;
            pbqp::Costs matrix(registers * registers, copy);
            for (size_t r = 0; r < registers; ++r)
                matrix[r * registers + r] = interferes(m, n) ? pbqp::kInfinity : 0;
            instance.graph.link(m, n, matrix);
        }
    }

    // The interval that ends first is simplicial among the remaining ones.
    ankerl::svector<size_t, 8> order(node_count);
    std::ranges::iota(order, size_t{0});
    std::ranges::sort(order, [&](size_t a, size_t b) { return intervals[a].second < intervals[b].second; });
    instance.peo_position.resize(node_count);
    for (size_t i = 0; i < node_count; ++i)
        instance.peo_position[order[i]] = i;
    return instance;
}

void peo_priority_always_colors_chordal_graphs() {
    std::mt19937_64 random{61};
    for (int trial = 0; trial < 500; ++trial) {
        const IntervalInstance instance = random_interval_instance(random, 6 + static_cast<size_t>(trial % 30));
        const pbqp::Solution solution   = pbqp::solve(instance.graph, instance.peo_position);
        CHECK(solution.feasible());
        CHECK(solution.cost == pbqp::evaluate(instance.graph, solution.choices));
    }
}

void statistics_count_every_node() {
    std::mt19937_64 random{67};
    const IntervalInstance instance = random_interval_instance(random, 30);
    const pbqp::Statistics stats    = pbqp::solve(instance.graph, instance.peo_position).statistics;
    CHECK(stats.r0 + stats.r1 + stats.r2 + stats.rn + stats.single_option == instance.graph.node_count());

    pbqp::Graph path;
    for (int i = 0; i < 5; ++i)
        path.add_node({1, 2});
    for (pbqp::NodeId n = 1; n < 5; ++n)
        path.link(n - 1, n, {0, 3, 3, 0});
    const pbqp::Statistics path_stats = pbqp::solve(path).statistics;
    CHECK(path_stats.rn == 0);
    CHECK(path_stats.r0 + path_stats.r1 + path_stats.r2 == 5);
}

void infeasible_instances_are_reported() {
    pbqp::Graph graph;
    for (int i = 0; i < 3; ++i)
        graph.add_node({0, 0});
    const pbqp::Costs interference = {pbqp::kInfinity, 0, 0, pbqp::kInfinity};
    for (pbqp::NodeId n = 0; n < 3; ++n)
        graph.link(n, (n + 1) % 3, interference);
    CHECK(!pbqp::solve(graph).feasible());
}

void rm_merges_implied_neighbour() {
    // Figure 2 of Buchwald et al. 2011: coloring U implies V, and merging V into U leaves only exact reductions.
    constexpr auto inf = pbqp::kInfinity;
    pbqp::Graph graph;
    const pbqp::NodeId u             = graph.add_node({0, 0});
    const pbqp::NodeId v             = graph.add_node({0, 0});
    const pbqp::NodeId w             = graph.add_node({0, 0, 0});
    const pbqp::NodeId y             = graph.add_node({0, 0, 0});
    const pbqp::NodeId a             = graph.add_node({0, 0, 0});
    const pbqp::NodeId b             = graph.add_node({0, 0, 0});
    const pbqp::Costs interference33 = {inf, 0, 0, 0, inf, 0, 0, 0, inf};
    const pbqp::Costs affinity23     = {0, 1, 1, 1, 0, 1};
    graph.link(u, w, {inf, 0, 0, 0, inf, 0});
    graph.link(u, v, {inf, 0, 0, inf});
    graph.link(v, w, {inf, 0, 0, 0, inf, 0});
    graph.link(v, y, affinity23);
    graph.link(u, a, affinity23);
    graph.link(w, b, interference33);
    graph.link(a, y, interference33);
    graph.link(y, b, interference33);
    graph.link(a, b, interference33);

    const std::array<size_t, 6> priority = {5, 0, 4, 1, 2, 3};
    const pbqp::Solution solution        = pbqp::solve(graph, priority);
    CHECK(solution.statistics.rm >= 1);
    CHECK(solution.cost == pbqp::solve_exhaustive(graph)->cost);
}

void evaluate_prices_an_assignment() {
    pbqp::Graph graph;
    static_cast<void>(graph.add_node({1, 2}));

    CHECK(pbqp::evaluate(graph, pbqp::Choices{1}) == 2);
}

void exhaustive_refuses_large_instances() {
    pbqp::Graph graph;
    for (int i = 0; i < 4; ++i)
        graph.add_node({0, 1, 2});
    CHECK(!pbqp::solve_exhaustive(graph, 80).has_value());
    CHECK(pbqp::solve_exhaustive(graph, 81).has_value());
}

void find_edge_works_in_both_directions() {
    pbqp::Graph graph;
    const pbqp::NodeId a = graph.add_node({1, 2});
    const pbqp::NodeId b = graph.add_node({1, 2});

    graph.link(a, b, {0, 0, 0, 0});
    CHECK(graph.find_edge(a, b) != pbqp::Graph::npos);
    CHECK(graph.find_edge(b, a) != pbqp::Graph::npos);
}

void accessors_and_setters_round_trip() {
    pbqp::Graph graph;
    const pbqp::NodeId m = graph.add_node({1, 2});
    const pbqp::NodeId n = graph.add_node({3, 4, 5});
    const pbqp::EdgeId e = graph.link(m, n, {0, 1, 2, 3, 4, 5});

    const auto [src, dst] = graph.edge(e);
    CHECK(src == m);
    CHECK(dst == n);
    CHECK(graph.incident_edges(n).size() == 1);
    CHECK(graph.edge_cost(e, 1, 2) == 5);

    graph.set_node_costs(m, std::array<pbqp::Cost, 2>{7, 8});
    graph.set_edge_costs(e, std::array<pbqp::Cost, 6>{6, 5, 4, 3, 2, 1});
    CHECK(graph.node_costs(m)[1] == 8);
    CHECK(graph.edge_cost(e, 1, 2) == 1);
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
    CHECK(graph.find_edge(0, 2) == pbqp::Graph::npos);

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
    for (size_t m = 0; m < 6; ++m) {
        for (size_t n = m + 1; n < 6; ++n) {
            if (cost(random) < 5) continue;
            std::ranges::generate(matrix, next_cost);
            graph.link(m, n, matrix);
        }
    }

    const pbqp::Solution sequential = *pbqp::solve_exhaustive(graph, 1'000'000, 1);
    const pbqp::Solution parallel   = *pbqp::solve_exhaustive(graph, 1'000'000, 4);
    CHECK(sequential.choices == parallel.choices);
    CHECK(sequential.cost == parallel.cost);
}

void wide_option_vectors_stay_exact() {
    std::mt19937_64 random{41};
    std::uniform_int_distribution<int> cost{0, 100};
    const auto next_cost = [&] { return static_cast<pbqp::Cost>(cost(random)); };

    for (const auto options : {size_t{17}, size_t{33}}) {
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
        CHECK(pbqp::solve(pair).cost == pbqp::solve_exhaustive(pair)->cost);

        // A triangle has no low-degree node, so RII contracts a wide matrix.
        pbqp::Graph triangle;
        ankerl::svector<pbqp::NodeId, 4> nodes;
        for (int i = 0; i < 3; ++i) {
            std::ranges::generate(values, next_cost);
            nodes.emplace_back(triangle.add_node(values));
        }
        for (size_t i = 0; i < nodes.size(); ++i) {
            std::ranges::generate(matrix, next_cost);
            triangle.link(nodes[i], nodes[(i + 1) % nodes.size()], matrix);
        }
        CHECK(pbqp::solve(triangle).cost == pbqp::solve_exhaustive(triangle)->cost);
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
    for (size_t m = 0; m < 7; ++m) {
        for (size_t n = m + 1; n < 7; ++n) {
            if (cost(random) < 5) continue;
            std::ranges::generate(matrix, next_cost);
            graph.link(m, n, matrix);
        }
    }

    const pbqp::Solution heuristic = pbqp::solve(graph);
    const pbqp::Solution optimal   = *pbqp::solve_exhaustive(graph, 1'000'000, 1);

    constexpr size_t thread_count = 4;
    std::array<pbqp::Solution, thread_count> heuristics;
    std::array<pbqp::Solution, thread_count> optima;
    std::array<std::thread, thread_count> workers;

    for (size_t i = 0; i < thread_count; ++i) {
        workers[i] = std::thread([&, i] {
            heuristics[i] = pbqp::solve(graph);
            optima[i]     = *pbqp::solve_exhaustive(graph, 1'000'000, 2);
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
    heuristic_is_optimal_on_random_trees(0, 10, 0);
    heuristic_is_optimal_on_random_trees(-1000, 1000, 0);
    heuristic_is_optimal_on_random_trees(-1000, 1000, 20);
    heuristic_is_optimal_on_random_trees(-1'000'000, 1'000'000, 20);
    long_path_with_large_costs_stays_exact();
    rn_charges_its_edges_to_the_neighbours();
    heuristic_result_is_always_valid();
    negative_costs_reward_coalescing();
    infinity_forbids_same_option();
    infinite_sums_do_not_wrap();
    pruning_makes_dense_graphs_exact();
    heuristic_handles_forbidden_options_on_dense_graphs();
    peo_priority_always_colors_chordal_graphs();
    statistics_count_every_node();
    infeasible_instances_are_reported();
    rm_merges_implied_neighbour();
    evaluate_prices_an_assignment();
    exhaustive_refuses_large_instances();
    find_edge_works_in_both_directions();
    accessors_and_setters_round_trip();
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
