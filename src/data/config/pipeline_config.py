from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PipelineConfig:
    data_amount: int
    data_format: str
    annotation_format: str
    augment_amount: int
    resize_size: Tuple[int, int]
    worker_ips: List[str]
    port: int
    world_size: int
    use_fake_data: bool
    log_level: str
