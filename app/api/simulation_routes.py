"""P1 Simulation API routes."""
from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.models.simulation import (
    AskRequest,
    AskResponse,
    PredictionReport,
    SimulationConfig,
    SimulationRequest,
    SimulationStatus,
    SimulationStatusResponse,
)
from app.services.simulation import (
    _llm_call,
    _simulations,
    create_simulation_id,
    get_simulation,
    run_simulation,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["simulation"])


# ── Helpers ──────────────────────────────────────────────────

def _get_world_model(document_id: str) -> dict:
    """Retrieve world model from P0 in-memory store."""
    try:
        from app.api.routes import _world_models
        wm = _world_models.get(document_id)
    except (ImportError, AttributeError):
        wm = None

    if wm is None:
        raise HTTPException(
            status_code=404,
            detail=f"World model not found for document_id={document_id}. "
                   "Run extraction first via POST /api/v1/extract.",
        )

    if hasattr(wm, "model_dump"):
        return wm.model_dump()
    if hasattr(wm, "dict"):
        return wm.dict()
    return wm


# ── POST /api/simulations ───────────────────────────────────

@router.post("/simulations", status_code=202)
async def create_simulation(
    request: SimulationRequest,
    background_tasks: BackgroundTasks,
):
    world_model = _get_world_model(request.document_id)

    sim_id = create_simulation_id()
    config = SimulationConfig(
        simulation_id=sim_id,
        **request.model_dump(),
    )

    background_tasks.add_task(run_simulation, config, world_model)
    return {"simulation_id": sim_id, "status": "running"}


# ── GET /api/simulations/{id} ───────────────────────────────

@router.get(
    "/simulations/{simulation_id}",
    response_model=SimulationStatusResponse,
)
async def get_simulation_status(simulation_id: str):
    sim = get_simulation(simulation_id)
    if sim is None:
        raise HTTPException(status_code=404, detail="Simulation not found")

    return SimulationStatusResponse(
        simulation_id=sim.simulation_id,
        status=sim.status,
        current_round=len(sim.rounds),
        total_rounds=sim.config.max_rounds,
        rounds_completed=sim.rounds,
    )


# ── GET /api/simulations/{id}/report ─────────────────────────

@router.get(
    "/simulations/{simulation_id}/report",
    response_model=PredictionReport,
)
async def get_simulation_report(simulation_id: str):
    sim = get_simulation(simulation_id)
    if sim is None:
        raise HTTPException(status_code=404, detail="Simulation not found")

    if sim.status == SimulationStatus.RUNNING:
        raise HTTPException(status_code=202, detail="Simulation still running")
    if sim.status == SimulationStatus.FAILED:
        raise HTTPException(status_code=500, detail="Simulation failed")
    if sim.final_report is None:
        raise HTTPException(status_code=404, detail="Report not yet generated")

    return sim.final_report


# ── POST /api/simulations/{id}/ask ───────────────────────────

@router.post(
    "/simulations/{simulation_id}/ask",
    response_model=AskResponse,
)
async def ask_simulation(simulation_id: str, request: AskRequest):
    sim = get_simulation(simulation_id)
    if sim is None:
        raise HTTPException(status_code=404, detail="Simulation not found")

    if sim.status not in (
        SimulationStatus.COMPLETED,
        SimulationStatus.TERMINATED_EARLY,
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot ask: simulation status is {sim.status.value}",
        )

    # Build context from rounds
    ctx = []
    for r in sim.rounds:
        ctx.append(f"第 {r.round_number} 轮：")
        ctx.append(f"  场景：{r.scene_description[:300]}")
        for a in r.actor_actions:
            ctx.append(f"  {a.actor_name}: {a.action[:200]}")
        ctx.append(f"  裁决：{r.adjudication.round_summary[:300]}")

    if sim.final_report:
        ctx.append(f"\n预测报告摘要：{sim.final_report.summary}")

    context = "\n".join(ctx)

    system_prompt = """你是推演结果的解读助手。
基于完整的推演过程和预测报告，回答用户的追问。

输出要求（严格 JSON）：
{
  "answer": "回答（200-500字）",
  "evidence": ["依据1（引用具体轮次和角色行动）", ...]
}

规则：
- 回答必须基于推演过程中的实际数据
- 引用具体的轮次、角色行动、变量变化作为依据
- 如果推演中没有相关信息，明确说明
- 只输出合法 JSON"""

    user_prompt = f"""推演过程：
{context}

用户追问：{request.question}

请回答。"""

    result = await _llm_call(system_prompt, user_prompt, temperature=0.2)
    return AskResponse(**result)
