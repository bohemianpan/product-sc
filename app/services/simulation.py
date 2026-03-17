"""P1 Simulation engine — core loop.

Narrator + Actor(parallel) + Adjudicator, round-based.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from openai import AsyncOpenAI

from app.config import Settings
from app.models.simulation import (
    ActorAction,
    ActorState,
    AdjudicationResult,
    Intervention,
    NarratorOutput,
    PredictionReport,
    RelationshipState,
    RoundResult,
    SimulationConfig,
    SimulationResult,
    SimulationStatus,
    TimelineEvent,
    WorldState,
)

logger = logging.getLogger(__name__)

# ── In-memory store (MVP) ────────────────────────────────────
_simulations: dict[str, SimulationResult] = {}


def get_simulation(sim_id: str) -> SimulationResult | None:
    return _simulations.get(sim_id)


# ── LLM helper ───────────────────────────────────────────────

async def _llm_call(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Call LLM and return parsed JSON dict.

    Reuses the same Qwen/OpenAI-compatible config from P0.
    """
    if settings is None:
        settings = Settings()

    api_key = settings.get_api_key()
    base_url = settings.get_base_url()
    model = settings.get_model()

    if not api_key:
        raise ValueError("No API key configured. Set LLM_API_KEY in .env")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    if not raw:
        raise ValueError("LLM returned empty response")

    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    return json.loads(cleaned)


# ── Init world state from P0 world model ────────────────────

def init_world_state(world_model: dict, config: SimulationConfig) -> WorldState:
    """Convert P0 WorldModel dict to P1 runtime WorldState."""
    actors_raw = world_model.get("actors", [])
    actors = [
        ActorState(
            name=a.get("name", "Unknown"),
            role=a.get("role", a.get("type", "unknown")),
            description=a.get("description", ""),
        )
        for a in actors_raw[: config.max_actors]
    ]

    relationships = [
        RelationshipState(
            source=r.get("from", r.get("source", "")),
            target=r.get("to", r.get("target", "")),
            relation=r.get("type", r.get("relation", "")),
            description=r.get("description", ""),
        )
        for r in world_model.get("relationships", [])
    ]

    variables: dict[str, str] = {}
    for v in world_model.get("variables", []):
        name = v.get("name", v.get("variable", ""))
        value = v.get("current_value", v.get("value", "unknown"))
        if name:
            variables[name] = str(value)

    timeline = [
        TimelineEvent(
            time=t.get("date", t.get("time", "")),
            event=t.get("event", ""),
            actors_involved=t.get("actors_involved", []),
        )
        for t in world_model.get("timeline", [])
    ]

    return WorldState(
        actors=actors,
        relationships=relationships,
        variables=variables,
        timeline=timeline,
        round_number=0,
    )


# ── Narrator step ────────────────────────────────────────────

async def narrator_step(
    world_state: WorldState,
    previous_actions: list[ActorAction],
    interventions: list[Intervention],
    config: SimulationConfig,
) -> NarratorOutput:
    actors_desc = "\n".join(
        f"- {a.name} ({a.role}): {a.description}" for a in world_state.actors
    )
    vars_desc = "\n".join(
        f"- {k}: {v}" for k, v in world_state.variables.items()
    ) or "无"

    prev_summary = ""
    if previous_actions:
        prev_summary = "上一轮各角色行动：\n" + "\n".join(
            f"- {a.actor_name}: {a.action}" for a in previous_actions
        )

    intervention_text = ""
    if interventions:
        intervention_text = "本轮新增事件（必须融入场景）：\n" + "\n".join(
            f"- {i.event}" for i in interventions
        )

    system_prompt = """你是一个推演场景的叙事者（Narrator）。
你的职责是：基于当前世界状态和上一轮各角色行动，描述本轮场景，推进时间线。

输出要求（严格 JSON）：
{
  "scene_description": "本轮场景描述（200-500字）",
  "world_state_updates": {"变量名": "新值", ...},
  "narrator_notes": "给各角色的额外提示"
}

规则：
- 保持时间推进逻辑一致
- 如有干预事件，自然融入场景
- 不要替角色做决策
- 场景描述要具体，有细节
- 只输出合法 JSON"""

    user_prompt = f"""预测问题：{config.question}
场景类型：{config.scenario_type.value}
时间范围：{config.time_horizon}
当前轮次：第 {world_state.round_number} 轮（共 {config.max_rounds} 轮）

角色列表：
{actors_desc}

当前变量：
{vars_desc}

{prev_summary}

{intervention_text}

请生成本轮场景描述。"""

    result = await _llm_call(system_prompt, user_prompt, temperature=0.3)
    return NarratorOutput(**result)


# ── Actor step ────────────────────────────────────────────────

async def actor_step(
    actor: ActorState,
    scene: NarratorOutput,
    world_state: WorldState,
) -> ActorAction:
    relevant_rels = [
        r for r in world_state.relationships
        if r.source == actor.name or r.target == actor.name
    ]
    relationships_desc = ""
    if relevant_rels:
        relationships_desc = "你的关系：\n" + "\n".join(
            f"- 与 {r.target if r.source == actor.name else r.source}: {r.relation}"
            for r in relevant_rels
        )

    memory_text = ""
    if actor.memory:
        memory_text = "你的历史行动记录：\n" + "\n".join(
            f"- {m}" for m in actor.memory[-5:]
        )

    system_prompt = f"""你是 {actor.name}，角色定位是 {actor.role}。
角色描述：{actor.description}

你的行为要符合你的角色设定和历史行为模式。
基于当前场景，决定你的行动。

输出要求（严格 JSON）：
{{
  "actor_name": "{actor.name}",
  "action": "你本轮的具体行动（100-300字）",
  "reasoning": "决策理由（50-150字）",
  "stance_changes": {{"态度维度": "新立场"}} 或 null,
  "signals": ["释放的信号1", "信号2"]
}}

规则：
- 从你的角色视角出发
- 行动要具体、可执行
- 不能做超出你角色能力范围的事
- 只输出合法 JSON"""

    user_prompt = f"""当前场景：
{scene.scene_description}

{scene.narrator_notes}

{relationships_desc}

{memory_text}

当前你的立场：{json.dumps(actor.stance, ensure_ascii=False) if actor.stance else "无特别立场"}

请决定你本轮的行动。"""

    result = await _llm_call(system_prompt, user_prompt, temperature=0.5)
    result["actor_name"] = actor.name
    return ActorAction(**result)


# ── Adjudicator step ─────────────────────────────────────────

async def adjudicator_step(
    world_state: WorldState,
    scene: NarratorOutput,
    actor_actions: list[ActorAction],
    config: SimulationConfig,
) -> AdjudicationResult:
    actions_desc = "\n".join(
        f"- {a.actor_name}: {a.action} (理由: {a.reasoning})"
        for a in actor_actions
    )
    vars_desc = "\n".join(
        f"- {k}: {v}" for k, v in world_state.variables.items()
    ) or "无"

    system_prompt = """你是推演的中立裁判（Adjudicator）。
你的职责是：收集所有角色的行动，判定实际结果，更新世界变量，判断是否终止。

输出要求（严格 JSON）：
{
  "world_state_updates": {"变量名": "更新后的值", ...},
  "conflict_resolutions": ["冲突裁决说明1", ...],
  "round_summary": "本轮总结（200-400字）",
  "should_terminate": false,
  "termination_reason": null
}

规则：
- 中立客观
- 如果两个角色的行动互斥，必须裁决谁成功
- 变量更新要合理
- 如果局势已经明朗或无法继续推进，设置 should_terminate 为 true
- 只输出合法 JSON"""

    user_prompt = f"""预测问题：{config.question}
当前轮次：第 {world_state.round_number} 轮（共 {config.max_rounds} 轮）

场景：{scene.scene_description}

各角色行动：
{actions_desc}

当前变量：
{vars_desc}

请裁决本轮结果。"""

    result = await _llm_call(system_prompt, user_prompt, temperature=0.1)
    return AdjudicationResult(**result)


# ── State mutation helpers ───────────────────────────────────

def apply_adjudication(world_state: WorldState, adj: AdjudicationResult) -> None:
    for key, value in adj.world_state_updates.items():
        world_state.variables[key] = value
    world_state.timeline.append(
        TimelineEvent(
            time=f"Round {world_state.round_number}",
            event=adj.round_summary[:200],
        )
    )


def update_actor_memory(
    world_state: WorldState, actions: list[ActorAction]
) -> None:
    action_map = {a.actor_name: a for a in actions}
    for actor in world_state.actors:
        if actor.name in action_map:
            a = action_map[actor.name]
            actor.memory.append(
                f"Round {world_state.round_number}: {a.action}"
            )
            if a.stance_changes:
                actor.stance.update(a.stance_changes)


# ── Report generation ────────────────────────────────────────

async def generate_report(
    config: SimulationConfig,
    rounds: list[RoundResult],
    world_state: WorldState,
) -> PredictionReport:
    rounds_summary = ""
    for r in rounds:
        rounds_summary += f"\n--- 第 {r.round_number} 轮 ---\n"
        rounds_summary += f"场景：{r.scene_description[:300]}\n"
        for a in r.actor_actions:
            rounds_summary += f"  {a.actor_name}: {a.action[:200]}\n"
        rounds_summary += f"裁决：{r.adjudication.round_summary[:300]}\n"

    final_vars = "\n".join(
        f"- {k}: {v}" for k, v in world_state.variables.items()
    ) or "无"

    system_prompt = """你是预测报告生成器。
基于完整的推演过程，生成结构化预测报告。

输出要求（严格 JSON）：
{
  "summary": "结论摘要（200-500字）",
  "most_likely_path": "最可能的发展路径（200-400字）",
  "turning_points": ["转折点1", "转折点2", ...],
  "key_variables": ["变量1: 最终状态", ...],
  "risks": ["风险1", "风险2", ...],
  "opportunities": ["机会1", ...],
  "recommended_actions": ["建议1", "建议2", ...],
  "evidence_refs": ["依据1", ...]
}

规则：
- 结论必须有推演依据
- 标注关键转折点
- 风险和机会要具体可操作
- 建议行动要务实
- 只输出合法 JSON"""

    user_prompt = f"""预测问题：{config.question}
场景类型：{config.scenario_type.value}
时间范围：{config.time_horizon}

推演过程：
{rounds_summary}

最终变量状态：
{final_vars}

请生成预测报告。"""

    result = await _llm_call(system_prompt, user_prompt, temperature=0.2)
    return PredictionReport(**result)


# ── Main entry ───────────────────────────────────────────────

async def run_simulation(
    config: SimulationConfig, world_model: dict
) -> SimulationResult:
    """Run a full simulation and return the result."""
    sim = SimulationResult(
        simulation_id=config.simulation_id,
        config=config,
        rounds=[],
        status=SimulationStatus.RUNNING,
    )
    _simulations[config.simulation_id] = sim

    try:
        world_state = init_world_state(world_model, config)
        rounds: list[RoundResult] = []

        for round_num in range(1, config.max_rounds + 1):
            world_state.round_number = round_num
            logger.info(
                "Simulation %s: round %d/%d",
                config.simulation_id, round_num, config.max_rounds,
            )

            # Interventions for this round
            interventions = [
                i for i in config.interventions
                if i.trigger_round == round_num
            ]

            # Narrator
            previous_actions = rounds[-1].actor_actions if rounds else []
            scene = await narrator_step(
                world_state, previous_actions, interventions, config,
            )

            # Apply narrator variable updates
            for k, v in scene.world_state_updates.items():
                world_state.variables[k] = v

            # Actors (parallel)
            actor_tasks = [
                actor_step(actor, scene, world_state)
                for actor in world_state.actors
            ]
            actor_actions = list(await asyncio.gather(*actor_tasks))

            # Adjudicator
            adjudication = await adjudicator_step(
                world_state, scene, actor_actions, config,
            )

            # Update world state
            apply_adjudication(world_state, adjudication)
            update_actor_memory(world_state, actor_actions)

            # Record round
            round_result = RoundResult(
                round_number=round_num,
                scene_description=scene.scene_description,
                actor_actions=actor_actions,
                adjudication=adjudication,
                world_state_snapshot=world_state.model_copy(deep=True),
            )
            rounds.append(round_result)
            sim.rounds = rounds

            if adjudication.should_terminate:
                logger.info(
                    "Simulation %s: terminated early at round %d — %s",
                    config.simulation_id, round_num,
                    adjudication.termination_reason,
                )
                break

        # Generate report
        report = await generate_report(config, rounds, world_state)
        sim.final_report = report
        sim.status = SimulationStatus.COMPLETED
        logger.info(
            "Simulation %s: completed, %d rounds",
            config.simulation_id, len(rounds),
        )

    except Exception:
        logger.exception("Simulation %s: failed", config.simulation_id)
        sim.status = SimulationStatus.FAILED

    _simulations[config.simulation_id] = sim
    return sim


def create_simulation_id() -> str:
    return f"sim_{uuid.uuid4().hex[:12]}"
