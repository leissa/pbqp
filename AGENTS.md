# pbqp

C++23/CMake template for a PBQP (Partitioned Boolean Quadratic Programming) solver: a dependency-free library, a demo CLI, and a CTest binary.

## Build & test

Presets are the entry point; Debug is the default:

```sh
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

| Task | Command |
| ---- | ------- |
| Build | `cmake --build --preset debug` (also `release`, `asan`, `tsan`) |
| Tests | `ctest --preset debug --output-on-failure` |
| One test binary | `./build/debug/pbqp_tests` |
| Demo | `./build/debug/pbqp_demo --nodes 20 --options 3 --edges 40` (`--in`/`--out FILE` read/write JSON, `--dot FILE` writes Graphviz) |
| Format | `clang-format -i include/pbqp/*.h src/pbqp/*.cpp apps/*.cpp tests/*.cpp` |

Warnings are on (`-Wall -Wextra -Wpedantic -Wconversion -Wshadow`, `-Wshadow=local` on GCC) and the build is kept warning-free; `-DPBQP_WARNINGS_AS_ERRORS=ON` enforces that.

## Layout

- `include/pbqp/` public headers; `src/pbqp/` implementations, mirroring the include tree.
- `apps/` demo CLI, `tests/` dependency-free CTest binary.
- `submodules/fe` is a public dependency (`fe`); the public headers use `fe::Vector` for storage and `fe::View`/`fe::Span` for spans.
- `Graph` stores cost vectors/matrices with stable ids and at most one edge per node pair (`add_edge_costs` sums onto an existing edge in either direction); storage stays private (read-only spans, `edge(e)` endpoints, `set_node_costs`/`set_edge_costs`) so equal edge matrices are hash-consed into one refcounted copy (`ankerl::unordered_dense` keyed by hash, collisions chained in the pool); `Cost` is `std::int32_t`, `CostSum` totals are `std::int64_t`; `kInfinity` marks forbidden entries and makes any total `kInfiniteSum`.
- `solve` reduces a normalized `int32` working copy (non-negative, saturating at `kInfinity`, all-zero edges dropped) whose edges borrow the graph's matrices until their first write, which copies into an `fe::Arena` as plain arrays; transient matrices live in a second, rolled-back scratch arena, and RII/RM results are allocated last so they can become the edge's matrix or be rolled back with degree-bucketed worklists: forbidden options are deleted (tracked back to original indices), single-option nodes and RI/RII are reduced exactly, RM merges implied neighbours right before RN, RN commits to the locally best option, ordered by the optional `rn_priority` (PEO positions for SSA-based register allocation); it back-substitutes in reverse reduction order; infeasible solutions carry `conflicts`, the nodes whose choice hits a `kInfinity` entry. `solve_exhaustive` is the reference for tiny instances.
- `read_json`/`write_json` speak the pbqp-papa format: node `index` is the node id, edge `cost` is row-major with `source` as the row, `"inf"` (also `"∞"`, `"INF"`) is `kInfinity`; version 1 stores costs as strings, version 2 (written) as integers; `peo[0]` is decided first, so it maps to the highest `rn_priority`; the reader is built on `fe::Lexer`/`fe::Parser`, keeps string escapes verbatim, recovers at anchored `}`/`]`, and throws everything it collected once parsing is done.
- Reduction kernels are plain 32-bit add/min loops that the compiler vectorizes; on x86 ELF `target_clones` adds x86-64-v4/AVX2 runtime dispatch (off under ThreadSanitizer).

## Comments

Comments are scarce. The default is **no comment**.

Comment only when the code itself cannot reasonably express the information.

- Comment **why**, not what the code does.
- Prefer a better name, structure, or API over a comment.
- Keep comments to **one short sentence**, normally one line.
- When a comment spans multiple lines, use **one complete sentence per line**. Do not wrap a single sentence across multiple lines merely to fit a line-length limit.
- A comment should convey one fact only: an invariant, non-obvious constraint, algorithmic reason, or important external reference.
- Do not explain the implementation, summarize a function, or provide a narrative of its control flow.
- Match the comment density and brevity of the surrounding code. **Never increase comment density.**
- Do not add documentation-style prose, introductions, conclusions, or motivational/explanatory language.
- Do not use rhetorical contrasts such as `"X" -> "Y"`, `"instead of X"`, or `"from X to Y"` to explain an optimization.
- Do not add comments describing the change itself ("now handles X", "renamed from Y"); that belongs in the commit message.
- Do not add banner or section-header comments.
- Do not add a comment if deleting it would leave the code equally correct and understandable.
- A comment that restates the code is worse than no comment:
  ```cpp
  vec.push_back(x); // BAD: "put x into the vector"
  ```

**Hard limit:** Do not write multi-line comments unless the user explicitly asks for documentation or the comment is required to document a non-obvious invariant that cannot be stated briefly.

Before adding a comment, ask:
1. Is this information necessary?
2. Is it already apparent from the code or names?
3. Can it be expressed in one short sentence?
If the answer to 1 or 3 is no, do not add the comment.
