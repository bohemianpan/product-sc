"""P1 Simulation data models."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ScenarioType(str, Enum):
    PUBLIC_OPINION = "public_opinion"
    POLICY = "policy"
    NARRATIVE = "narrative"
    CUSTOM = "custom"


class SimulationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    TERMINATED_EARLY = "terminated_early"
    FAILED = "failed"


# ── Intervention ──────────────────────────────────────────────

class Intervention(BaseModel):
    trigger_round: int
    event: str
    affected_variables: list[str] = []


# ── Simulation config (request) ──────────────────────────────

class SimulationRequest(BaseModel):
    document_id: str
    question: str
    scenario_type: ScenarioType = ScenarioType.CUSTOM
    time_horizon: str = "7d"
    max_rounds: int = Field(default=5, le=5, ge=1)
    max_actors: int = Field(default=8, le=8, ge=1)
    interventions: list[Intervention] = []


class SimulationConfig(SimulationRequest):
    simulation_id: str


# ── World state (runtime) ────────────────────────────────────

class ActorState(BaseModel):
    name: str
    role: str
    description: str
    memory: list[str] = []
    stance: dict[str, str] = {}


class RelationshipState(BaseModel):
    source: str
    target: str
    relation: str
    description: str = ""


class TimelineEvent(BaseModel):
    time: str
    event: str
    actors_involved: list[str] = []


class WorldState(BaseModel):
    actors: list[ActorState]
    relationships: list[RelationshipState]
    variables: dict[str, str] = {}
    timeline: list[TimelineEvent] = []
    round_number: int = 0


# ── Per-round outputs ────────────────────────────────────────

class NarratorOutput(BaseModel):
    scene_description: str
    world_state_updates: dict[str, str] = {}
    narrator_notes: str = ""


class ActorAction(BaseModel):
    actor_name: str
    action: str
    reasoning: str
    stance_changes: Optional[dict[str, str]] = None
    signals: list[str] = []


class AdjudicationResult(BaseModel):
    world_state_updates: dict[str, str] = {}
    conflict_resolutions: list[str] = []
    round_summary: str
    should_terminate: bool = False
    termination_reason: Optional[str] = None


class RoundResult(BaseModel):
    round_number: int
    scene_description: str
    actor_actions: list[ActorAction]
    adjudication: AdjudicationResult
    world_state_snapshot: WorldState


# ── Final report ─────────────────────────────────────────────

class PredictionReport(BaseModel):
    summary: str
    most_likely_path: str
    turning_points: list[str] = []
    key_variables: list[str] = []
    risks: list[str] = []
    opportunities: list[str] = []
    recommended_actions: list[str] = []
    evidence_refs: list[str] = []


# ── Simulation result ────────────────────────────────────────

class SimulationResult(BaseModel):
    simulation_id: str
    config: SimulationConfig
    rounds: list[RoundResult] = []
    final_report: Optional[PredictionReport] = None
    status: SimulationStatus = SimulationStatus.PENDING


# ── API response helpers ─────────────────────────────────────

class SimulationStatusResponse(BaseModel):
    simulation_id: str
    status: SimulationStatus
    current_round: int = 0
    total_rounds: int = 5
    rounds_completed: list[RoundResult] = []


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    evidence: list[str] = []
