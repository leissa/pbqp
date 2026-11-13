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
| Demo | `./build/debug/pbqp_demo --nodes 20 --options 3 --edges 40` |
| Format | `clang-format -i include/pbqp/*.hpp src/pbqp/*.cpp apps/*.cpp tests/*.cpp` |

Warnings are on (`-Wall -Wextra -Wpedantic -Wconversion -Wshadow`) and the build is kept warning-free; `-DPBQP_WARNINGS_AS_ERRORS=ON` enforces that.

## Layout

- `include/pbqp/` public headers; `src/pbqp/` implementations, mirroring the include tree.
- `apps/` demo CLI, `tests/` dependency-free CTest binary.
- `submodules/svector` is a public dependency (`ankerl::svector`) exported by `pbqp::pbqp`.
- `Graph` stores cost vectors/matrices with stable ids and at most one edge per node pair; `Cost` is `std::int16_t` with saturating sums, `CostSum` totals are `std::int64_t`.
- `solve` applies RI/RII exactly and RN heuristically, then back-substitutes in reverse reduction order; `solve_exhaustive` is the reference for tiny instances.
- RI/RII kernels use saturating `int16` SIMD with runtime AVX-512BW/AVX2 dispatch on x86, NEON on ARM, and a scalar fallback.

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
