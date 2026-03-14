import torch


def get_device() -> torch.device:
    """디바이스 자동 감지: CUDA > MPS > CPU 우선순위로 선택."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("[Device] CPU")
    return device
