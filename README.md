# Poster v2

Layout (36 x 24 inch, PTC 2026 horizontal template):

| | Column 1 | Column 2 | Column 3 |
|---|---|---|---|
| Row 1 | Past / Present | 01 `torch.cond`: skip optimizer if gradient is non-finite | The captured CUDA graph (IF/ELSE node) |
| Row 2 | Support matrix (cond / while_loop / switch) | 02 `torch.switch`: token-count buckets + fixed-width embedding output | 03 EP=8 lossless MoE: grouped GEMM + `torch.while_loop` |

```sh
/opt/ssd/miniforge3/envs/pt23/bin/python poster_PTC2026/v2/examples.py      # needs a GPU
/opt/ssd/miniforge3/envs/pt23/bin/python poster_PTC2026/v2/build_poster.py  # poster.pptx -> poster.pdf
```

`examples.py` holds the functions shown on the poster. The build checks that the
displayed code has the same AST as `examples.py`, then runs the same PDF checks as
v1 (page size, embedded fonts, text inside its boxes, highlights, links).

What `examples.py` verifies on PyTorch main (91a0eda, CUDA 12.9, RTX 3090):
`safe_sgd` and `embedding_bag` give correct results under
`torch.compile(..., backend="cudagraphs")` across replays with changing inputs.
`moe_decode` is checked with `backend="aot_eager"`; it switches to
`backend="cudagraphs"` automatically once the SWITCH lowering (PR #189461) is
in the build.

The matrix icons (tick, construction barrier) are vector shapes, not emoji, so
they print cleanly. `poster.pptx` is overwritten on every build.

The MoE panel shows excerpts from the original `ep8_moe_backpressure.py`
(revision `89ee54e33a1d99c9ed5d69383377ad98f3cf9df0`). Its full source is
included alongside the poster. The build compares the displayed statements
against that source by AST. The distributed GPU example was not rerun locally.

The embedding example selects capacities 32, 128, 512, or 4096 using a GPU
scalar. Input ID storage is fixed at 4096 slots, but each branch gathers and
reduces only its selected prefix. All branches return a vector of shape `[D]`.
GPU verification covers zero tokens, bucket boundaries, and maximum capacity.
These GPU checks are provided in `examples.py`; they were not run on this Mac.
