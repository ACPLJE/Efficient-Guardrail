"""System plugin interface (SPEC 4.2). Plugins emit events; the harness measures."""
from __future__ import annotations

import abc
import itertools
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from ag.data.serde import JsonMixin
from ag.data.types import Request
from ag.harness.context import RunContext
from ag.harness.events import Event


@dataclass(frozen=True)
class SystemInfo(JsonMixin):
    system_id: str
    mode: Literal["live", "offline", "external"]
    engine: str | None
    attn_backend: str | None
    patch_applied: bool
    patch_sha: str | None
    shim_active: bool
    model_sha: Mapping[str, str]
    placement: Mapping[str, str]
    flags: Mapping[str, Any]


@dataclass(frozen=True)
class SystemConfig:
    """Resolved system YAML (SPEC 8.2) plus the experiment-level knobs the system needs."""
    system_id: str
    impl: str
    flags: Mapping[str, Any]
    exp: Mapping[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.flags.get(key, default)


class System(abc.ABC):
    system_id: ClassVar[str]

    @abc.abstractmethod
    def setup(self, cfg: SystemConfig, ctx: RunContext) -> None: ...

    @abc.abstractmethod
    def run(self, req: Request) -> Iterator[Event]:
        """Blocking generator of one request's events (thread-safe)."""

    def run_many(self, reqs: Iterable[Request]) -> Iterator[Event]:
        return itertools.chain.from_iterable(self.run(r) for r in reqs)

    def reset(self) -> None:
        pass

    def teardown(self) -> None:
        pass

    @abc.abstractmethod
    def info(self) -> SystemInfo: ...
