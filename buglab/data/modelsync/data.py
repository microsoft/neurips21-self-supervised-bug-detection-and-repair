from typing import NamedTuple, Optional


class ModelSyncData(NamedTuple):
    """Data structure used by model syncing client and server."""

    model: Optional[bytes]
    model_version: float
    params: Optional[bytes]
    params_version: float
