#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <algorithm>
#include <format>
#include <fstream>
#include <print>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <ankerl/svector.h>

#include "pbqp/dot.h"
#include "pbqp/graph.h"
#include "pbqp/solver.h"

namespace {

struct Settings {
    size_t nodes       = 12;
    size_t options     = 4;
    size_t edges       = 24;
    std::uint64_t seed = 42;
    std::string dot;
};

void print_usage(std::string_view program) {
    std::println("usage: {} [--nodes N] [--options K] [--edges E] [--seed S] [--dot FILE]", program);
    std::println("  builds a random PBQP instance, solves it, and (for small instances)");
    std::println("  compares the heuristic result against brute force");
    std::println("  --dot writes the instance and the heuristic assignment as Graphviz DOT");
}

template<class T>
T parse_number(std::string_view name, std::string_view text) {
    T value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size())
        throw std::invalid_argument(std::format("{} expects a non-negative integer, got '{}'", name, text));
    return value;
}

Settings parse(int argc, char** argv) {
    Settings settings;

    for (int i = 1; i < argc; ++i) {
        const std::string_view argument = argv[i];
        const auto next                 = [&](std::string_view name) {
            if (i + 1 >= argc) throw std::invalid_argument(std::format("missing value for {}", name));
            return argv[++i];
        };

        if (argument == "--nodes")
            settings.nodes = parse_number<size_t>(argument, next(argument));
        else if (argument == "--options")
            settings.options = parse_number<size_t>(argument, next(argument));
        else if (argument == "--edges")
            settings.edges = parse_number<size_t>(argument, next(argument));
        else if (argument == "--seed")
            settings.seed = parse_number<std::uint64_t>(argument, next(argument));
        else if (argument == "--dot")
            settings.dot = next(argument);
        else if (argument == "-h" || argument == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument(std::format("unknown argument: {}", argument));
        }
    }

    if (settings.nodes == 0 || settings.options == 0)
        throw std::invalid_argument("--nodes and --options must be positive");
    return settings;
}

pbqp::Graph random_instance(const Settings& settings) {
    std::mt19937_64 random{settings.seed};
    std::uniform_int_distribution<int> cost{0, 10};

    pbqp::Graph graph;
    ankerl::svector<pbqp::Cost, 4> node_costs(settings.options);
    for (size_t i = 0; i < settings.nodes; ++i) {
        std::ranges::generate(node_costs, [&] { return static_cast<pbqp::Cost>(cost(random)); });
        graph.add(node_costs);
    }

    const size_t max_edges = settings.nodes % 2 == 0 ? settings.nodes / 2 * (settings.nodes - 1)
                                                     : (settings.nodes - 1) / 2 * settings.nodes;
    const size_t edges     = std::min(settings.edges, max_edges);

    ankerl::svector<pbqp::Cost, 4> edge_costs(settings.options * settings.options);
    const auto link = [&](pbqp::NodeId m, pbqp::NodeId n) {
        std::ranges::generate(edge_costs, [&] { return static_cast<pbqp::Cost>(cost(random)); });
        graph.link(m, n, edge_costs);
    };

    if (2 * edges > max_edges) {
        ankerl::svector<std::pair<pbqp::NodeId, pbqp::NodeId>, 8> pairs;
        pairs.reserve(max_edges);
        for (pbqp::NodeId m = 0; m < settings.nodes; ++m)
            for (pbqp::NodeId n = m + 1; n < settings.nodes; ++n)
                pairs.emplace_back(m, n);
        std::ranges::shuffle(pairs, random);
        for (size_t i = 0; i < edges; ++i)
            link(pairs[i].first, pairs[i].second);
    } else {
        // At most half of all pairs are taken, so rejection sampling needs fewer than two draws per edge.
        std::uniform_int_distribution<pbqp::NodeId> node{0, settings.nodes - 1};
        while (graph.edge_count() < edges) {
            const pbqp::NodeId m = node(random);
            const pbqp::NodeId n = node(random);
            if (m != n && graph.find_edge(m, n) == pbqp::Graph::npos) link(m, n);
        }
    }
    return graph;
}

void print_assignment(const pbqp::Graph& graph, const pbqp::Solution& solution) {
    std::print("assignment:");
    size_t column = 0;
    for (const auto n : graph.nodes()) {
        if (column != 0 && column % 10 == 0) std::print("\n           ");
        std::print(" n{}={}", n, solution.choices[n]);
        ++column;
    }
    std::println();
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Settings settings = parse(argc, argv);
        const pbqp::Graph graph = random_instance(settings);

        std::println("instance : {} nodes, {} edges, {} options per node", graph.node_count(), graph.edge_count(),
                     settings.options);

        const pbqp::Solution heuristic = pbqp::solve(graph);
        std::println("heuristic: cost {}", heuristic.cost);
        const pbqp::Statistics& stats = heuristic.statistics;
        std::println(
            "reduced  : R0 {}, R1 {}, R2 {}, RN {}, RM {}, single-option {}; {} forbidden options, {} dropped edges",
            stats.r0, stats.r1, stats.r2, stats.rn, stats.rm, stats.single_option, stats.forbidden_options,
            stats.dropped_edges);

        if (const auto optimal = pbqp::solve_exhaustive(graph, 2'000'000)) {
            std::print("brute    : cost {}", optimal->cost);
            if (heuristic.cost == optimal->cost) {
                std::println("  (heuristic matches the optimum)");
            } else {
                const auto gap       = static_cast<double>(heuristic.cost - optimal->cost);
                const double percent = 100.0 * gap / static_cast<double>(std::max<pbqp::CostSum>(optimal->cost, 1));
                std::println("  (heuristic is {:.1f}% above the optimum)", percent);
            }
        } else {
            std::println("brute    : skipped, instance too large for brute force");
        }

        print_assignment(graph, heuristic);

        if (!settings.dot.empty()) {
            std::ofstream dot{settings.dot};
            if (!dot) throw std::runtime_error(std::format("cannot open '{}'", settings.dot));
            pbqp::write_dot(dot, graph, heuristic.choices);
        }
        return 0;
    } catch (const std::exception& error) {
        std::println(stderr, "error: {}", error.what());
        return 1;
    }
}
