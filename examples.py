"""Runnable sources for poster v2: torch.cond / while_loop / switch in CUDA graphs.

The poster shows safe_sgd, embedding_bag, and moe_decode verbatim.
"""

import torch
from torch._higher_order_ops import switch


# 01 torch.cond -> IF/ELSE node: skip an optimizer step on non-finite gradients.
def safe_sgd(param, grad, lr):
    finite = torch.isfinite(grad).all()
    return torch.cond(
        finite,
        lambda p, g, rate: p - rate * g,
        lambda p, g, rate: p.clone(),
        (param, grad, lr),
    )


# 02 torch.while_loop -> WHILE node: reduce a runtime-length bag in fixed chunks.
def embedding_bag(table, ids, n):
    C = 4
    i = torch.zeros((), dtype=torch.int64, device=ids.device)
    acc = table.new_zeros(table.shape[1])

    def body(i, acc):
        pos = i + torch.arange(C, device=ids.device)
        rows = table[ids[pos.clamp_max(len(ids) - 1)]]
        return i + C, acc + (rows * (pos < n)[:, None]).sum(0)

    _, acc = torch.while_loop(lambda i, acc: i < n, body, (i, acc))
    return acc


# 03 torch.switch -> SWITCH node: a top-1 MoE decode step runs only the routed expert.
def moe_decode(x, router, w_up, w_down):
    expert = (x @ router).argmax()
    experts = [
        lambda x, e=e: torch.relu(x @ w_up[e]) @ w_down[e]
        for e in range(len(w_up))
    ]
    return switch(expert, experts, (x,))


def switch_node_available():
    # The SWITCH lowering comes with https://github.com/pytorch/pytorch/pull/189461.
    return hasattr(torch._C._CUDAGraph, "begin_capture_to_switch_node")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(0)

    # Each compiled function is captured once; later calls replay that graph.
    with torch.no_grad():
        step = torch.compile(safe_sgd, backend="cudagraphs")
        param = torch.tensor([1.0, 2.0], device=device)
        lr = torch.tensor(0.1, device=device)
        for values, expected in (
            ([0.5, -1.0], [0.95, 2.1]),
            ([0.5, -1.0], [0.95, 2.1]),
            ([torch.inf, 1.0], [1.0, 2.0]),
            ([0.5, -1.0], [0.95, 2.1]),
        ):
            grad = torch.tensor(values, device=device)
            updated = step(param, grad, lr)
            torch.testing.assert_close(updated, torch.tensor(expected, device=device))

        bag = torch.compile(embedding_bag, backend="cudagraphs")
        table = torch.arange(40, dtype=torch.float32, device=device).reshape(10, 4)
        ids = torch.tensor([1, 3, 2, 8, 0, 6, 4, 9], device=device)
        for n in (3, 3, 0, 8, 5):
            reduced = bag(table, ids, torch.tensor(n, device=device))
            torch.testing.assert_close(reduced, table[ids[:n]].sum(0))

        hidden, width, num_experts = 64, 256, 8
        router = torch.randn(hidden, num_experts, device=device)
        w_up = torch.randn(num_experts, hidden, width, device=device) / hidden**0.5
        w_down = torch.randn(num_experts, width, hidden, device=device) / width**0.5
        backend = "cudagraphs" if switch_node_available() else "aot_eager"
        decode = torch.compile(moe_decode, backend=backend)
        for _ in range(16):
            x = torch.randn(1, hidden, device=device)
            e = (x @ router).argmax()
            expected = torch.relu(x @ w_up[e]) @ w_down[e]
            torch.testing.assert_close(decode(x, router, w_up, w_down), expected)

    print("cond and while_loop passed with backend='cudagraphs'.")
    if switch_node_available():
        print("switch passed with backend='cudagraphs'.")
    else:
        print("switch passed with backend='aot_eager'; CUDA graph capture needs PR #189461.")


if __name__ == "__main__":
    main()
