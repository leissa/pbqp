#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <algorithm>
#include <format>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <ankerl/svector.h>

#include "pbqp/graph.hpp"
#include "pbqp/solver.hpp"

namespace {

struct Settings {
    std::size_t nodes   = 12;
    std::size_t options = 4;
    std::size_t edges   = 24;
    std::uint64_t seed  = 42;
};

void print_usage(std::string_view program) {
    std::cout << std::format("usage: {} [--nodes N] [--options K] [--edges E] [--seed S]\n", program)
              << "  builds a random PBQP instance, solves it, and (for small instances)\n"
              << "  compares the heuristic result against brute force\n";
}

auto parse(int argc, char** argv) -> Settings {
    Settings settings;

    for (int i = 1; i < argc; ++i) {
        const std::string_view argument = argv[i];
        const auto next                 = [&](std::string_view name) -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(std::format("missing value for {}", name));
            return argv[++i];
        };

        if (argument == "--nodes")
            settings.nodes = std::stoul(next(argument));
        else if (argument == "--options")
            settings.options = std::stoul(next(argument));
        else if (argument == "--edges")
            settings.edges = std::stoul(next(argument));
        else if (argument == "--seed")
            settings.seed = std::stoull(next(argument));
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

auto random_instance(const Settings& settings) -> pbqp::Graph {
    std::mt19937_64 random{settings.seed};
    std::uniform_real_distribution<double> cost{0.0, 10.0};

    pbqp::Graph graph;
    ankerl::svector<double, 4> node_costs(settings.options);
    for (std::size_t i = 0; i < settings.nodes; ++i) {
        std::ranges::generate(node_costs, [&] { return cost(random); });
        graph.add_node(node_costs);
    }

    ankerl::svector<std::pair<pbqp::NodeId, pbqp::NodeId>, 8> pairs;
    pairs.reserve(settings.nodes * (settings.nodes - 1) / 2);
    for (pbqp::NodeId u = 0; u < settings.nodes; ++u)
        for (pbqp::NodeId v = u + 1; v < settings.nodes; ++v)
            pairs.emplace_back(u, v);
    std::ranges::shuffle(pairs, random);

    ankerl::svector<double, 4> edge_costs(settings.options * settings.options);
    for (std::size_t i = 0; i < std::min(settings.edges, pairs.size()); ++i) {
        std::ranges::generate(edge_costs, [&] { return cost(random); });
        graph.add_edge(pairs[i].first, pairs[i].second, edge_costs);
    }
    return graph;
}

void print_assignment(const pbqp::Graph& graph, const pbqp::Solution& solution) {
    std::cout << "assignment:";
    std::size_t column = 0;
    for (const pbqp::NodeId v : graph.nodes()) {
        if (column != 0 && column % 10 == 0) std::cout << "\n           ";
        std::cout << std::format(" v{}={}", v, solution.choices[v]);
        ++column;
    }
    std::cout << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Settings settings = parse(argc, argv);
        const pbqp::Graph graph = random_instance(settings);

        std::cout << std::format("instance : {} nodes, {} edges, {} options per node\n", graph.node_count(),
                                 graph.edge_count(), settings.options);

        const pbqp::Solution heuristic = pbqp::solve(graph);
        std::cout << std::format("heuristic: cost {:.3f}\n", heuristic.cost);

        try {
            const pbqp::Solution optimal = pbqp::solve_exhaustive(graph, 2'000'000);
            std::cout << std::format("brute    : cost {:.3f}", optimal.cost);
            if (heuristic.cost <= optimal.cost + 1e-9) {
                std::cout << "  (heuristic matches the optimum)\n";
            } else {
                const double gap = 100.0 * (heuristic.cost - optimal.cost) / std::max(optimal.cost, 1e-9);
                std::cout << std::format("  (heuristic is {:.1f}% above the optimum)\n", gap);
            }
        } catch (const std::runtime_error&) { std::cout << "brute    : skipped, instance too large for brute force\n"; }

        print_assignment(graph, heuristic);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << std::format("error: {}\n", error.what());
        return 1;
    }
}
