# pbqp

[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue?style=flat-square&logo=cplusplus)](https://en.wikipedia.org/wiki/C%2B%2B#Standardization)
[![License](https://img.shields.io/github/license/leissa/pbqp?style=flat-square&color=blue&logo=opensourceinitiative&logoColor=white&label=License)](https://github.com/leissa/pbqp/blob/master/LICENSE)

[![Linux](https://img.shields.io/github/actions/workflow/status/leissa/pbqp/linux.yml?style=flat-square&logo=linux&label=Linux&logoColor=white&branch=master)](https://github.com/leissa/pbqp/actions/workflows/linux.yml?query=branch%3Amaster)
[![macOS](https://img.shields.io/github/actions/workflow/status/leissa/pbqp/macos.yml?style=flat-square&logo=apple&label=macOS&branch=master)](https://github.com/leissa/pbqp/actions/workflows/macos.yml?query=branch%3Amaster)
[![Windows](https://img.shields.io/github/actions/workflow/status/leissa/pbqp/windows.yml?style=flat-square&label=⊞%20Windows&branch=master)](https://github.com/leissa/pbqp/actions/workflows/windows.yml?query=branch%3Amaster)

A small, dependency-free starting point for a [PBQP](https://en.wikipedia.org/wiki/Partitioned_Boolean_Quadratic_Programming)
(Partitioned Boolean Quadratic Programming) solver in C++23.

PBQP is the optimisation problem behind, among other things, graph-colouring
style register allocators:

```
minimise   sum_v c_v(x_v)  +  sum_{uv in E} M_uv(x_u, x_v)
```

Each node picks one option (`x_v`), pays `c_v(x_v)`, and every edge pays a
matrix entry that depends on the options of both endpoints. The problem is
NP-hard in general, which is why real solvers use a reduction-based heuristic
and only fall back to something expensive when it pays off.

## Layout

```
CMakeLists.txt              library + demo + tests
CMakePresets.json           debug / release / asan / tsan presets
include/pbqp/graph.hpp      problem graph (cost vectors and matrices)
include/pbqp/solver.hpp     solver interface
include/pbqp/dot.hpp        Graphviz DOT export
src/pbqp/graph.cpp          graph storage, validation
src/pbqp/solver.cpp         reduction-based heuristic
src/pbqp/dot.cpp            DOT writer
apps/pbqp_demo.cpp          random instance generator and CLI
tests/test_pbqp.cpp         test binary (CTest)
submodules/                 git submodules: svector (used), doctest, unordered_dense
```

Small vectors (cost vectors, matrices, adjacency, solver snapshots) use
`ankerl::svector` from `submodules/svector`; the public headers depend on it, and
`pbqp::pbqp` exports its include directory.

## Build and run

After cloning, initialize the submodules: `git submodule update --init`.

```sh
cmake --preset debug
cmake --build --preset debug
ctest --preset debug

# random instance: 20 nodes, 3 options each, 40 edges
./build/debug/pbqp_demo --nodes 20 --options 3 --edges 40

# render it with the heuristic assignment highlighted
./build/debug/pbqp_demo --nodes 8 --options 3 --edges 12 --dot g.dot && dot -Tsvg g.dot -o g.svg
```

For sanitizer builds use `--preset asan` (address/UB) or `--preset tsan`
(threads). The library itself is just
`pbqp::pbqp`; link it from another CMake target with
`target_link_libraries(my_target PRIVATE pbqp::pbqp)`.

## Using the library

```cpp
#include "pbqp/graph.hpp"
#include "pbqp/solver.hpp"

pbqp::Graph graph;
const auto a = graph.add_node({1, 4});       // two options
const auto b = graph.add_node({0, 2});
graph.link(a, b, {
    0, 9,   // a = 0, b = 0 / a = 0, b = 1
    9, 0,   // a = 1, b = 0 / a = 1, b = 1
});

const pbqp::Solution solution = pbqp::solve(graph);
// solution.choices[a], solution.choices[b], solution.cost
```

Costs are `pbqp::Cost` (`std::int32_t`) and totals are `pbqp::CostSum`
(`std::int64_t`). `pbqp::kInfinity` forbids an option or option pair: any
assignment that uses one costs `pbqp::kInfiniteSum`. Negative entries are
ordinary rewards (register-allocation affinity edges).

`solve` returns a feasible assignment plus its cost under the original graph.
`solve_exhaustive` computes the true optimum for tiny instances, splitting the
search space across `thread_count` threads (0 = hardware concurrency).
`evaluate` prices any assignment you already have.

All solver functions are safe to call concurrently on a shared graph; a graph
must not be mutated while another thread reads it. `solve` works on its own
copy, so independent instances can be solved in parallel too.

## What the solver does

Nodes are reduced one at a time (lowest degree first) and the choices are
reconstructed in reverse order:

| Degree | Reduction | Exact? |
| ------ | --------- | ------ |
| 0      | pick the cheapest option | yes |
| 1      | RI: fold the edge into the neighbour | yes |
| 2      | RII: contract the node onto its two neighbours | yes |
| any, 1 option | fold the single option into the neighbours | yes |
| 3+     | RN: commit to the locally best option and charge its edges to the neighbours | no |
| 3+, before RN | RM: merge a neighbour whose option is implied, then R1 it | yes |

Because RI/RII are exact, the heuristic is optimal on trees. On graphs with
cycles a local heuristic (`rn_score`, the cost spread per incident edge) decides
which node is reduced with RN. Degree-bucketed worklists keep node selection
cheap, so reduction scales roughly linearly with the graph size.

Before and during reduction the solver normalizes its working copy: matrix
row and column minima move into the node vectors, node vectors are shifted to
a minimum of 0, and edges whose matrix becomes all zero are dropped. Costs
therefore stay small and relative, so 32-bit arithmetic saturates only when
option costs really differ by `kInfinity`, and dropped edges lower degrees.

Whenever a node's costs change, the solver also deletes its forbidden
(`kInfinity`) options. Nodes left with a single option are then reduced exactly
regardless of their degree. Back-substitution maps the remaining options back to
their original indices.

`solve` takes an optional `rn_priority` per node: among the nodes that need RN,
the highest priority is decided first. For SSA-based register allocation pass
each node's position in a perfect elimination order of the interference graph;
the exact reductions keep that order valid, so coloring cannot fail
(Buchwald, Zwinkau, Bersch: *SSA-based Register Allocation with PBQP*, CC 2011).
Without priorities RN falls back to degree and cost spread.
`Solution::feasible()` reports whether the assignment avoids every `kInfinity`
entry, and `Solution::statistics` counts each kind of reduction.

Graph ids are stable and removing a node/edge simply leaves a hole, so no
index bookkeeping is required while reducing.

The reduction kernels are plain 32-bit add/min loops that the compiler
vectorizes; on x86 ELF targets `target_clones` adds x86-64-v4 (AVX-512) and
AVX2 variants that are picked at runtime.

## Ideas for what to add

- Revisit option dominance: option `a` of `v` can be deleted exactly if
  `c(a) - c(b) >= sum_u max_y (M_u(b, y) - M_u(a, y))`. A first version helped
  little on random graphs (about 0.25% lower cost) but was 2.5-5x slower, and
  caching per-edge tables did not help because normalization keeps changing the
  edges. It may pay off on real instruction-selection graphs, ideally with a
  degree cap or as an opt-in.
- Better RN ordering (Briggs et al.), R0/R1 lookahead, or a simple local-search
  improvement after back-substitution.
- Merge parallel matrix edges internally (already handled) and add an
  optimal-RII variant for small node option counts.
- Import/export: DIMACS-like text format, or a DOT writer for debugging.
- An exact branch-and-bound/ILP solver behind the same `solve`-style interface.
- Swap the plain `std::vector` storage for `std::mdspan` views if you want to
  plug in sparse or blocked matrices.

## References

- Scholz et al., *Partitioned Boolean Quadratic Programming* (2007)
- Hames & Scholz, *Nearly Optimal Register Allocation with PBQP* (2006)
- Briggs, *Register Allocation via Graph Coloring* (PhD thesis, 1992) - RN/RI/RII reductions
- Buchwald & Zwinkau, *Instruction Selection by Graph Transformation* (CASES 2010)
- Buchwald, Zwinkau & Bersch, *SSA-based Register Allocation with PBQP* (CC 2011) - RM, PEO-ordered RN
