# Poster v2

Layout (36 x 24 inch, PTC 2026 horizontal template):

| | Column 1 | Column 2 | Column 3 |
|---|---|---|---|
| Row 1 | Support matrix (cond / while_loop / switch) | 01 `torch.cond`: skip non-finite optimizer steps | The captured CUDA graph (IF/ELSE node) |
| Row 2 | 02 `torch.while_loop`: embedding bag + WHILE node | 03 `torch.switch`: top-1 MoE decode + SWITCH node | The contract; Work Before General Use |

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
