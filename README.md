# Poster v2

Layout (36 x 24 inch, PTC 2026 horizontal template):

| | Column 1 | Column 2 | Column 3 |
|---|---|---|---|
| Row 1 | Past / Present / Further examples (QR) | 01 `torch.cond`: skip optimizer if gradient is non-finite | The captured CUDA graph (IF/ELSE node) |
| Row 2 | Support matrix (cond / while_loop / switch) | 02 `torch.switch`: token-count buckets + fixed-width embedding output | 03 EP=8 lossless MoE: grouped GEMM + `torch.while_loop` |

Column 1 runs through both rows; the row rule spans only columns 2 and 3.

```sh
python examples.py      # needs a GPU
python build_poster.py  # poster_rebuilt.pptx -> poster_rebuilt.pdf (+ previews)
```

`examples.py` holds the functions shown on the poster. The build checks that the
displayed code has the same AST as `examples.py`, then runs the same PDF checks as
v1 (page size, embedded fonts, text inside its boxes, highlights, links).

`examples.py` checks `safe_sgd` with `torch.compile(..., backend="cudagraphs")`
across replays with changing inputs. `embedding_bag` and `moe_decode` use
`backend="cudagraphs"` once the SWITCH lowering (PR #189461) is in the build,
and `backend="aot_eager"` otherwise.

The Further examples QR code (`examples_qr.png`) encodes
https://ibm.biz/controlflow-in-cuda-graphs, which redirects to `examples.py` on
this repository's `main` branch. The build lists the examples in
`FURTHER_EXAMPLES` and checks that each one is a function in `examples.py`.

The matrix icons (tick, construction barrier) are vector shapes, not emoji, so
they print cleanly. `poster_rebuilt.pptx` is overwritten on every build.

The MoE panel shows excerpts from the original `ep8_moe_backpressure.py`
(revision `89ee54e33a1d99c9ed5d69383377ad98f3cf9df0`). Its full source is
included alongside the poster. The build compares the displayed statements
against that source by AST. The distributed GPU example was not rerun locally.

The embedding example selects capacities 32, 128, 512, or 4096 using a GPU
scalar. Input ID storage is fixed at 4096 slots, but each branch gathers and
reduces only its selected prefix. All branches return a vector of shape `[D]`.
GPU verification covers zero tokens, bucket boundaries, and maximum capacity.
These GPU checks are provided in `examples.py`; they were not run on this Mac.
