"""Device selection helpers for experiment entry points."""

import os
import re
import subprocess
import warnings

import torch


def _run_nvidia_smi():
    """Return a short nvidia-smi summary, or None when unavailable."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,cuda_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines or None


def _cuda_version_tuple(version):
    if not version:
        return None
    match = re.match(r"^(\d+)\.(\d+)", str(version))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _minimum_driver_for_cuda(cuda_version):
    """Approximate minimum Linux NVIDIA driver for common CUDA toolkit builds."""
    version = _cuda_version_tuple(cuda_version)
    if version is None:
        return None
    major, minor = version
    minimums = {
        (12, 0): "525.60.13",
        (12, 1): "530.30.02",
        (12, 2): "535.54.03",
        (12, 3): "545.23.06",
        (12, 4): "550.54.14",
        (12, 5): "555.42.02",
        (12, 6): "560.28.03",
        (12, 8): "570.26",
    }
    candidates = [k for k in minimums if k <= (major, minor)]
    return minimums[max(candidates)] if candidates else None


def _cuda_status():
    """Probe CUDA once and return (available, diagnostic)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        available = torch.cuda.is_available()

    warnings_text = [str(item.message) for item in caught]
    diagnostic = {
        "torch_cuda": torch.version.cuda,
        "warnings": warnings_text,
        "nvidia_smi": _run_nvidia_smi(),
        "device_count": torch.cuda.device_count() if available else 0,
    }
    return available, diagnostic


def _format_cuda_diagnostic(diagnostic):
    lines = []
    torch_cuda = diagnostic.get("torch_cuda")
    if torch_cuda:
        lines.append(f"PyTorch CUDA build: {torch_cuda}")
        minimum_driver = _minimum_driver_for_cuda(torch_cuda)
        if minimum_driver:
            lines.append(f"Minimum NVIDIA driver for this CUDA build: >= {minimum_driver}")
    else:
        lines.append("PyTorch CUDA build: not available (CPU-only PyTorch wheel)")

    nvidia_smi = diagnostic.get("nvidia_smi")
    if nvidia_smi:
        lines.append("nvidia-smi GPU(s):")
        lines.extend(f"  - {line}" for line in nvidia_smi)
    else:
        lines.append("nvidia-smi GPU(s): unavailable")

    for warning in diagnostic.get("warnings", []):
        lines.append(f"CUDA probe warning: {warning}")
    return "\n".join(lines)


def resolve_device(requested="auto"):
    """Resolve the compute device for training.

    requested may be "auto", "cuda", "cpu", or a concrete CUDA device such as
    "cuda:0". The LEJEPA_DEVICE environment variable overrides "auto".
    """
    requested = (requested or "auto").lower()
    if requested == "auto":
        requested = os.environ.get("LEJEPA_DEVICE", "auto").lower()

    if requested == "cpu":
        return torch.device("cpu")

    if requested.startswith("cuda"):
        available, diagnostic = _cuda_status()
        if not available:
            details = _format_cuda_diagnostic(diagnostic)
            raise RuntimeError(
                f"CUDA device was requested ({requested}) but PyTorch cannot initialize CUDA.\n"
                f"{details}\n"
                "Fix by updating the NVIDIA driver or installing a PyTorch build whose CUDA "
                "runtime is compatible with the installed driver."
            )
        device = torch.device(requested)
        torch.cuda.set_device(device)
        print(f"CUDA: {torch.cuda.get_device_name(device)}")
        return device

    if requested != "auto":
        return torch.device(requested)

    available, diagnostic = _cuda_status()
    if available:
        device = torch.device("cuda")
        print(f"CUDA: {torch.cuda.get_device_name(device)}")
        return device

    print("CUDA unavailable; falling back to CPU.")
    print(_format_cuda_diagnostic(diagnostic))
    return torch.device("cpu")
