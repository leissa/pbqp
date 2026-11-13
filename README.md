# pbqp

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
CMakePresets.json           debug / release / asan presets
include/pbqp/graph.hpp      problem graph (cost vectors and matrices)
include/pbqp/solver.hpp     solver interface
src/pbqp/graph.cpp          graph storage, validation
src/pbqp/solver.cpp         reduction-based heuristic
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
```

For a sanitizer build use `--preset asan`. The library itself is just
`pbqp::pbqp`; link it from another CMake target with
`target_link_libraries(my_target PRIVATE pbqp::pbqp)`.

## Using the library

```cpp
#include "pbqp/graph.hpp"
#include "pbqp/solver.hpp"

pbqp::Graph graph;
const auto a = graph.add_node({1.0, 4.0});       // two options
const auto b = graph.add_node({0.0, 2.0});
graph.add_edge(a, b, {
    0.0, 9.0,   // a = 0, b = 0 / a = 0, b = 1
    9.0, 0.0,   // a = 1, b = 0 / a = 1, b = 1
});

const pbqp::Solution solution = pbqp::solve(graph);
// solution.choices[a], solution.choices[b], solution.cost
```

`solve` returns a feasible assignment plus its cost under the original graph.
`solve_exhaustive` computes the true optimum for tiny instances and is handy in
tests. `evaluate` prices any assignment you already have.

## What the solver does

Nodes are reduced one at a time (lowest degree first) and the choices are
reconstructed in reverse order:

| Degree | Reduction | Exact? |
| ------ | --------- | ------ |
| 0      | pick the cheapest option | yes |
| 1      | RI: fold the edge into the neighbour | yes |
| 2      | RII: contract the node onto its two neighbours | yes |
| 3+     | RN: remove the node, decide its option during back-substitution | no |

Because RI/RII are exact, the heuristic is optimal on trees. On graphs with
cycles a local heuristic (`rn_score`, the cost spread per incident edge) decides
which node is reduced with RN; the option itself is then chosen greedily from
the final choices of its neighbours.

Graph ids are stable and removing a node/edge simply leaves a hole, so no
index bookkeeping is required while reducing.

## Ideas for what to add

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
