"""P1 Simulation engine tests."""
from __future__ import annotations

import json
import pytest
from unittest.mock import patch

from app.models.simulation import (
    ActorAction,
    ActorState,
    AdjudicationResult,
    Intervention,
    NarratorOutput,
    PredictionReport,
    RoundResult,
    ScenarioType,
    SimulationConfig,
    SimulationResult,
    SimulationStatus,
    WorldState,
)
from app.services.simulation import (
    apply_adjudication,
    init_world_state,
    run_simulation,
    update_actor_memory,
)


# ── Fixtures ─────────────────────────────────────────────────

SAMPLE_WORLD_MODEL = {
    "actors": [
        {"name": "农夫山泉", "role": "market_leader", "description": "中国领先的瓶装水企业"},
        {"name": "消费者", "role": "群体", "description": "普通消费者群体"},
        {"name": "媒体", "role": "机构", "description": "新闻媒体和自媒体"},
    ],
    "relationships": [
        {"from": "农夫山泉", "to": "消费者", "type": "品牌-消费关系"},
        {"from": "媒体", "to": "农夫山泉", "type": "舆论监督"},
    ],
    "variables": [
        {"name": "品牌信任度", "current_value": "较高"},
        {"name": "舆情热度", "current_value": "正常"},
    ],
    "timeline": [
        {"date": "2024-01", "event": "农夫山泉发布年度报告"},
    ],
}


@pytest.fixture
def sample_config():
    return SimulationConfig(
        simulation_id="test_sim_001",
        document_id="doc_001",
        question="农夫山泉未来 72 小时舆情如何发展？",
        scenario_type=ScenarioType.PUBLIC_OPINION,
        time_horizon="72h",
        max_rounds=3,
        max_actors=8,
    )


# ── Unit: init_world_state ───────────────────────────────────

class TestInitWorldState:
    def test_basic_init(self, sample_config):
        ws = init_world_state(SAMPLE_WORLD_MODEL, sample_config)
        assert len(ws.actors) == 3
        assert ws.actors[0].name == "农夫山泉"
        assert ws.actors[0].role == "market_leader"
        assert len(ws.relationships) == 2
        assert ws.variables["品牌信任度"] == "较高"
        assert len(ws.timeline) == 1
        assert ws.round_number == 0

    def test_max_actors_limit(self, sample_config):
        sample_config.max_actors = 2
        ws = init_world_state(SAMPLE_WORLD_MODEL, sample_config)
        assert len(ws.actors) == 2

    def test_empty_world_model(self, sample_config):
        ws = init_world_state({}, sample_config)
        assert len(ws.actors) == 0
        assert len(ws.relationships) == 0
        assert len(ws.variables) == 0


# ── Unit: apply_adjudication ─────────────────────────────────

class TestApplyAdjudication:
    def test_updates_variables(self):
        ws = WorldState(
            actors=[], relationships=[],
            variables={"品牌信任度": "较高"},
            round_number=1,
        )
        adj = AdjudicationResult(
            world_state_updates={"品牌信任度": "下降", "舆情热度": "爆发"},
            round_summary="品牌信任度下降",
        )
        apply_adjudication(ws, adj)
        assert ws.variables["品牌信任度"] == "下降"
        assert ws.variables["舆情热度"] == "爆发"
        assert len(ws.timeline) == 1

    def test_appends_timeline(self):
        ws = WorldState(
            actors=[], relationships=[], timeline=[], round_number=2,
        )
        adj = AdjudicationResult(round_summary="第二轮总结")
        apply_adjudication(ws, adj)
        assert len(ws.timeline) == 1
        assert ws.timeline[0].time == "Round 2"


# ── Unit: update_actor_memory ────────────────────────────────

class TestUpdateActorMemory:
    def test_updates_memory_and_stance(self):
        ws = WorldState(
            actors=[
                ActorState(name="农夫山泉", role="企业", description=""),
                ActorState(name="媒体", role="机构", description=""),
            ],
            relationships=[], round_number=1,
        )
        actions = [
            ActorAction(
                actor_name="农夫山泉", action="发布声明",
                reasoning="稳定舆情",
                stance_changes={"态度": "积极回应"},
            ),
            ActorAction(
                actor_name="媒体", action="跟踪报道",
                reasoning="新闻价值",
            ),
        ]
        update_actor_memory(ws, actions)
        assert len(ws.actors[0].memory) == 1
        assert "发布声明" in ws.actors[0].memory[0]
        assert ws.actors[0].stance["态度"] == "积极回应"
        assert len(ws.actors[1].memory) == 1


# ── Mock LLM responses ──────────────────────────────────────

MOCK_NARRATOR = {
    "scene_description": "农夫山泉面临舆情压力，消费者开始关注水源地问题。",
    "world_state_updates": {},
    "narrator_notes": "各角色注意舆情动向。",
}

MOCK_ACTOR = {
    "action": "发布回应声明",
    "reasoning": "需要稳定品牌形象",
    "stance_changes": None,
    "signals": ["官方回应"],
}

MOCK_ADJUDICATOR = {
    "world_state_updates": {"舆情热度": "上升"},
    "conflict_resolutions": [],
    "round_summary": "各方反应激烈，舆情热度上升。",
    "should_terminate": False,
    "termination_reason": None,
}

MOCK_REPORT = {
    "summary": "农夫山泉舆情在 72 小时内经历了升温、扩散、回落三个阶段。",
    "most_likely_path": "舆情在 24 小时内达到峰值后逐步回落。",
    "turning_points": ["媒体跟进报道导致舆情扩散", "官方声明发布后热度回落"],
    "key_variables": ["品牌信任度: 略有下降", "舆情热度: 回落"],
    "risks": ["二次舆情爆发", "竞品趁机营销"],
    "opportunities": ["危机公关提升品牌形象"],
    "recommended_actions": ["持续监测舆情", "准备后续声明"],
    "evidence_refs": ["Round 1: 媒体跟进报道", "Round 2: 官方声明"],
}


# ── Integration: full simulation (mocked LLM) ───────────────

class TestRunSimulation:
    @pytest.mark.asyncio
    async def test_full_run(self, sample_config):
        call_count = 0

        async def mock_llm(system_prompt, user_prompt, temperature=0.3, settings=None):
            nonlocal call_count
            call_count += 1
            if "叙事者" in system_prompt or "Narrator" in system_prompt:
                return MOCK_NARRATOR
            elif "裁判" in system_prompt or "Adjudicator" in system_prompt:
                return MOCK_ADJUDICATOR
            elif "预测报告" in system_prompt:
                return MOCK_REPORT
            else:
                resp = MOCK_ACTOR.copy()
                return resp

        with patch("app.services.simulation._llm_call", side_effect=mock_llm):
            result = await run_simulation(sample_config, SAMPLE_WORLD_MODEL)

        assert result.status == SimulationStatus.COMPLETED
        assert len(result.rounds) == 3
        assert result.final_report is not None
        assert "农夫山泉" in result.final_report.summary

        for r in result.rounds:
            assert r.round_number >= 1
            assert len(r.actor_actions) == 3
            assert r.adjudication.round_summary != ""

        # 3 rounds × (1 narrator + 3 actors + 1 adjudicator) + 1 report = 16
        assert call_count == 16

    @pytest.mark.asyncio
    async def test_early_termination(self, sample_config):
        adj_count = 0

        async def mock_llm(system_prompt, user_prompt, temperature=0.3, settings=None):
            nonlocal adj_count
            if "叙事者" in system_prompt:
                return MOCK_NARRATOR
            elif "裁判" in system_prompt:
                adj_count += 1
                resp = MOCK_ADJUDICATOR.copy()
                if adj_count >= 2:
                    resp["should_terminate"] = True
                    resp["termination_reason"] = "局势已明朗"
                return resp
            elif "预测报告" in system_prompt:
                return MOCK_REPORT
            else:
                return MOCK_ACTOR.copy()

        with patch("app.services.simulation._llm_call", side_effect=mock_llm):
            result = await run_simulation(sample_config, SAMPLE_WORLD_MODEL)

        assert result.status == SimulationStatus.COMPLETED
        assert len(result.rounds) == 2

    @pytest.mark.asyncio
    async def test_intervention_injection(self, sample_config):
        sample_config.interventions = [
            Intervention(trigger_round=2, event="竞品发布负面营销"),
        ]
        narrator_prompts = []

        async def mock_llm(system_prompt, user_prompt, temperature=0.3, settings=None):
            if "叙事者" in system_prompt:
                narrator_prompts.append(user_prompt)
                return MOCK_NARRATOR
            elif "裁判" in system_prompt:
                return MOCK_ADJUDICATOR
            elif "预测报告" in system_prompt:
                return MOCK_REPORT
            else:
                return MOCK_ACTOR.copy()

        with patch("app.services.simulation._llm_call", side_effect=mock_llm):
            result = await run_simulation(sample_config, SAMPLE_WORLD_MODEL)

        assert result.status == SimulationStatus.COMPLETED
        assert "竞品发布负面营销" in narrator_prompts[1]
        assert "竞品发布负面营销" not in narrator_prompts[0]


# ── Model validation ─────────────────────────────────────────

class TestModels:
    def test_config_validation(self):
        config = SimulationConfig(
            simulation_id="t", document_id="d", question="q?",
            max_rounds=5, max_actors=8,
        )
        assert config.max_rounds == 5

    def test_config_max_rounds_limit(self):
        with pytest.raises(Exception):
            SimulationConfig(
                simulation_id="t", document_id="d", question="q?",
                max_rounds=10,
            )

    def test_report_serialization(self):
        report = PredictionReport(
            summary="test", most_likely_path="path",
            turning_points=["tp1"], risks=["r1"],
        )
        data = report.model_dump()
        assert data["summary"] == "test"
        assert len(data["turning_points"]) == 1
