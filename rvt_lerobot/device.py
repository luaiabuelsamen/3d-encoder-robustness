"""Pick the compute device, loudly, and explain the trap if it is sprung.

**Never replace PYTHONPATH when running anything in this repository.** This host
has two PyTorch installations:

    ~/projects/clean_env/pytorch/torch          source build, CUDA 12.6  -- works
    ~/.local/lib/python3.10/site-packages/torch wheel,        CUDA 13.0  -- does not

The driver is 12.6, so the wheel's CUDA runtime is too new for it and every
`torch.cuda.is_available()` returns False with the message "the NVIDIA driver on
your system is too old" -- which is exactly backwards and sends you to check the
driver.

The shell profile already exports a PYTHONPATH that puts
`~/projects/clean_env/pytorch` first, and that is the only reason the working
build wins. So `PYTHONPATH=. python ...` breaks CUDA by *clobbering* that value,
and `env -u PYTHONPATH python ...` breaks it by removing it. Either leave the
variable alone and let scripts put the repo on `sys.path` themselves -- which is
what they do, and what the Makefile relies on -- or extend it with
`PYTHONPATH=.:$PYTHONPATH`. Cost of learning this the slow way, and then of
learning it backwards first: two false starts.

The GPU is also shared with other sessions on this machine, so a failed probe is
worth retrying rather than dying on.
"""

from __future__ import annotations

import os
import sys


def _torch_provenance() -> str:
    import torch

    return f"torch {torch.__version__} (CUDA {torch.version.cuda}) from {torch.__file__}"


def pick_device(allow_cpu: bool = False, wait_s: float = 600.0, poll_s: float = 20.0) -> str:
    """Return "cuda", waiting for a busy GPU, or raise with the likely cause."""
    import time

    import torch

    deadline = time.time() + wait_s
    waited = False
    while True:
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")  # force context creation now, while we can
            if waited:
                print("  GPU free, continuing", flush=True)
            return "cuda"
        if time.time() >= deadline:
            break
        if not waited:
            print(f"  GPU unavailable; retrying for up to {wait_s:.0f}s", flush=True)
            print(f"  {_torch_provenance()}", flush=True)
            waited = True
        time.sleep(poll_s)

    if allow_cpu or os.environ.get("RVT_ALLOW_CPU"):
        print(f"  falling back to CPU. {_torch_provenance()}", flush=True)
        return "cpu"

    hint = ""
    if os.environ.get("PYTHONPATH"):
        hint = (
            "\n  PYTHONPATH is set to "
            f"{os.environ['PYTHONPATH']!r}. On this host that alone selects the "
            "wrong torch wheel (CUDA 13.0 against a 12.6 driver). Unset it and "
            "let the scripts put the repo on sys.path themselves."
        )
    raise RuntimeError(
        f"CUDA is not available after waiting {wait_s:.0f}s.\n  {_torch_provenance()}"
        f"{hint}\n  Otherwise another process may hold the GPU: "
        "`ps -eo pid,etime,cmd | grep python`. Pass --allow-cpu to proceed anyway."
    )
