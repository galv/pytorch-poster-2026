"""Runnable sources for poster v2: torch.cond / while_loop / switch in CUDA graphs.

The poster shows safe_sgd and the bucketed embedding_bag function.
"""

import torch
from torch._higher_order_ops import switch


# 01 torch.cond -> IF/ELSE node: skip an optimizer step on non-finite gradients.
def safe_sgd(param, grad, lr):
    finite = torch.isfinite(grad).all()

    def update():
        param.sub_(lr * grad)

    torch.cond(finite, update, lambda: None, ())


# 02 torch.switch: process only the selected token-count bucket.
def embedding_bag(table, ids, n):
    caps = (32, 128, 512, 4096)
    bucket = sum((n > c).to(torch.int32)
                 for c in caps[:-1])

    def reduce(cap):
        pos = torch.arange(cap, device=ids.device)
        rows = table[ids[:cap]]
        return (rows * (pos < n)[:, None]).sum(0)

    branches = [lambda c=c: reduce(c) for c in caps]
    return switch(bucket, branches, ())

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
        param_address = param.data_ptr()
        lr = torch.tensor(0.1, device=device)
        for values, expected in (
            ([0.5, -1.0], [0.95, 2.1]),
            ([0.5, -1.0], [0.9, 2.2]),
            ([torch.inf, 1.0], [0.9, 2.2]),
            ([0.5, -1.0], [0.85, 2.3]),
        ):
            grad = torch.tensor(values, device=device)
            step(param, grad, lr)
            assert param.data_ptr() == param_address
            torch.testing.assert_close(param, torch.tensor(expected, device=device))

        backend = "cudagraphs" if switch_node_available() else "aot_eager"
        bag = torch.compile(embedding_bag, backend=backend)
        table = torch.arange(40, dtype=torch.float32, device=device).reshape(10, 4)
        ids = torch.arange(4096, device=device) % len(table)
        for n in (0, 1, 32, 33, 80, 128, 129, 512, 513, 4096):
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

    print("cond passed with backend='cudagraphs'.")
    print(f"bucketed embedding_bag passed with backend={backend!r}.")
    if switch_node_available():
        print("switch passed with backend='cudagraphs'.")
    else:
        print("switch passed with backend='aot_eager'; CUDA graph capture needs PR #189461.")


if __name__ == "__main__":
    main()
