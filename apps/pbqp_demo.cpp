#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <algorithm>
#include <format>
#include <fstream>
#include <iostream>
#include <print>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

#include <fe/cli.h>
#include <fe/error.h>
#include <fe/vector.h>

#include "pbqp/dot.h"
#include "pbqp/graph.h"
#include "pbqp/json.h"
#include "pbqp/solver.h"

namespace {

struct Settings {
    size_t nodes       = 12;
    size_t options     = 4;
    size_t edges       = 24;
    std::uint64_t seed = 42;
    std::string in;
    std::string out;
    std::string dot;
};

pbqp::Instance read_instance(const std::string& path) {
    std::ifstream in{path};
    if (!in) throw std::runtime_error(std::format("cannot open '{}'", path));
    return pbqp::read_json(in, path);
}

pbqp::Graph random_instance(const Settings& settings) {
    std::mt19937_64 random{settings.seed};
    std::uniform_int_distribution<int> cost{0, 10};

    pbqp::Graph graph;
    fe::Vector<pbqp::Cost, 4> node_costs(settings.options);
    for (size_t i = 0; i < settings.nodes; ++i) {
        std::ranges::generate(node_costs, [&] { return static_cast<pbqp::Cost>(cost(random)); });
        graph.add(node_costs);
    }

    const size_t max_edges = settings.nodes % 2 == 0 ? settings.nodes / 2 * (settings.nodes - 1)
                                                     : (settings.nodes - 1) / 2 * settings.nodes;
    const size_t edges     = std::min(settings.edges, max_edges);

    fe::Vector<pbqp::Cost, 4> edge_costs(settings.options * settings.options);
    const auto link = [&](pbqp::NodeId m, pbqp::NodeId n) {
        std::ranges::generate(edge_costs, [&] { return static_cast<pbqp::Cost>(cost(random)); });
        graph.link(m, n, edge_costs);
    };

    if (2 * edges > max_edges) {
        fe::Vector<std::pair<pbqp::NodeId, pbqp::NodeId>, 8> pairs;
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
        while (graph.num_edges() < edges) {
            const pbqp::NodeId m = node(random);
            const pbqp::NodeId n = node(random);
            if (m != n && graph.edge(m, n) == pbqp::Graph::npos) link(m, n);
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
    fe::term::resolve_mode();

    try {
        bool show_help = false;
        Settings settings;

        // clang-format off
        auto cli = fe::Cli("pbqp_demo", "Solves a random or given PBQP instance and compares the heuristic against brute force on small ones.")
            .help(show_help)
            .grp("Random instance")
            .opt(settings.nodes  , "num" , "", "--nodes"  , "Number of nodes.")
            .opt(settings.options, "num" , "", "--options", "Options per node.")
            .opt(settings.edges  , "num" , "", "--edges"  , "Number of edges; capped at a complete graph.")
            .opt(settings.seed   , "num" , "", "--seed"   , "Seed of the random generator.")
            .grp("Files")
            .opt(settings.in     , "file", "", "--in"     , "Reads the instance from JSON instead.")
            .opt(settings.out    , "file", "", "--out"    , "Writes the instance as JSON.")
            .opt(settings.dot    , "file", "", "--dot"    , "Writes the instance and the heuristic assignment as Graphviz DOT.");
        // clang-format on

        if (auto err = cli.parse(argc, argv)) throw std::invalid_argument(*err);

        if (show_help) {
            std::cout << cli;
            return EXIT_SUCCESS;
        }

        if (settings.nodes == 0 || settings.options == 0)
            throw std::invalid_argument("--nodes and --options must be positive");

        const pbqp::Instance instance
            = settings.in.empty() ? pbqp::Instance{random_instance(settings), {}} : read_instance(settings.in);
        const pbqp::Graph& graph = instance.graph;

        if (settings.in.empty())
            std::println("instance : {} nodes, {} edges, {} options per node", graph.num_nodes(), graph.num_edges(),
                         settings.options);
        else
            std::println("instance : {} nodes, {} edges from {}", graph.num_nodes(), graph.num_edges(), settings.in);

        const pbqp::Solution heuristic = pbqp::solve(graph, instance.rn_priority);
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

        if (!settings.out.empty()) {
            std::ofstream out{settings.out};
            if (!out) throw std::runtime_error(std::format("cannot open '{}'", settings.out));
            pbqp::write_json(out, graph, instance.rn_priority);
        }

        if (!settings.dot.empty()) {
            std::ofstream dot{settings.dot};
            if (!dot) throw std::runtime_error(std::format("cannot open '{}'", settings.dot));
            pbqp::write_dot(dot, graph, heuristic.choices);
        }
        return EXIT_SUCCESS;
    } catch (const fe::Error::Bail& bail) {
        std::cerr << bail;
        return EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::println(stderr, "error: {}", error.what());
        return EXIT_FAILURE;
    }
}
