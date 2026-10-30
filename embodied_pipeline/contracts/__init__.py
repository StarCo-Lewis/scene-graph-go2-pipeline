"""Architecture metadata only; legacy wire/schema contracts stay unchanged."""

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Tuple, Optional


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    responsibility: str
    llm_calls: str


MODULES = (
    ModuleSpec("perception", "Read supplied scene relations; preserve provenance", "none added"),
    ModuleSpec("planning", "Task decomposition and object binding", "existing calls"),
    ModuleSpec("instruction_skills", "Skill retrieval and action matrix proposal", "existing calls"),
    ModuleSpec("validation", "Existing protocol, reference and state checks", "none added"),
    ModuleSpec("execution", "Existing symbolic proxy; future robot interface", "none"),
    ModuleSpec("errors", "Existing rejection and infrastructure boundaries", "none added"),
    ModuleSpec("memory", "Symbolic state/history transaction and audit persistence", "none"),
)


class Proposer(Protocol):
    def __call__(self, args, folder, encoder) -> Tuple[dict, Any, Any]: ...


class Executor(Protocol):
    def __call__(self, plan, alignment, library, nodes, state, compact) -> dict: ...


@dataclass(frozen=True)
class EpisodeServices:
    """Explicit test/Oracle seams; defaults preserve the existing runtime.

    No Oracle implementation or reference plan is supplied or enabled here.
    A future intervention must be named and audited as a separate experiment.
    """
    proposer: Optional[Proposer] = None
    executor: Optional[Executor] = None
    scene_loader: Optional[Callable] = None

