import torch.distributed as dist
from loguru import logger
import os
import sys

def print_rank0(*args, **kwargs):
    if not dist.is_available():
        print(*args, **kwargs)
        return

    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)
    
def get_rank() -> int:
    # Works before and after torch.distributed initialization
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))

def setup_logger(enabled: bool = True) -> None:
    logger.remove()

    if not enabled:
        return

    log_level = "INFO" if get_rank() == 0 else "WARNING"

    logger.add(
        sys.stdout,
        level=log_level,
        enqueue=True,
        backtrace=False,
        diagnose=False,
        colorize=True,
        format=(
            "<green>[rank={rank}]</green> "
            "<cyan>{{time:YYYY-MM-DD HH:mm:ss}}</cyan> | "
            "<level>{{level}}</level> | "
            "<level>{{message}}</level>"
        ).format(rank=get_rank()),
    )