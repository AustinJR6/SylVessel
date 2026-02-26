from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from core.prompt_engineer import PromptEngineer
from core.adaptation.preference_engine import PreferenceEngine
from core.emotion.emotion_model import EmotionModel
from core.identity.identity_manager import IdentityManager
from core.inference.inference_engine import InferenceEngine
from core.memory.memory_repository import SupabaseMemoryRepository
from core.memory.memory_types import MemoryRecord, MemoryType, RetrievalQuery
from core.telemetry.vitals_engine import VitalsEngine
from tools.analytics.reporting import Reporting
from tools.outreach.contact_resolver import ContactResolver
from tools.outreach.email_generator import EmailGenerator
from tools.outreach.lead_collector import LeadCollector
from tools.tool_contract import ToolRequest, ToolResult
from tools.tool_registry import ToolRegistry
from tools.tool_router import ToolRouter
from tools.trading.trading_interface import TradingInterface

logger = logging.getLogger(__name__)


class Brain:
    """Companion brain layer orchestrating memory, inference, telemetry, and tools."""

    def __init__(
        self,
        *,
        inference: InferenceEngine,
        memory: SupabaseMemoryRepository,
        emotion: EmotionModel,
        identity: IdentityManager,
        preferences: PreferenceEngine,
        vitals: VitalsEngine,
        router: ToolRouter,
    ):
        self.inference = inference
        self.memory = memory
        self.emotion = emotion
        self.identity = identity
        self.preferences = preferences
        self.vitals = vitals
        self.router = router
        self.prompt_engineer = PromptEngineer()
        self.emotional_history: List[str] = []
        self.turn_count = 0

    @classmethod
    def create_default(cls, mode: str = "claude") -> "Brain":
        registry = ToolRegistry()
        lead_collector = LeadCollector()
        resolver = ContactResolver()
        emailer = EmailGenerator()
        trading = TradingInterface()
        reporting = Reporting()

        def outreach_handler(req: ToolRequest) -> ToolResult:
            if req.action == "generate_leads":
                data = lead_collector.generate_leads(
                    product=str(req.parameters.get("product", "manifest")),
                    limit=int(req.parameters.get("limit", 5)),
                )
                return ToolResult.success(data, f"{len(data.get('leads', []))} leads generated")
            if req.action == "resolve_contact":
                data = resolver.resolve_contact(str(req.parameters.get("company_name", "")))
                return ToolResult.success(data, "Contact lookup completed")
            if req.action == "generate_email":
                data = emailer.generate_email(
                    company=str(req.parameters.get("company", "")),
                    contact=str(req.parameters.get("contact", "")),
                    value_prop=str(req.parameters.get("value_prop", "solar workflow automation")),
                )
                return ToolResult.success(data, "Email draft created")
            return ToolResult.error(f"Unsupported outreach action: {req.action}")

        def trading_handler(req: ToolRequest) -> ToolResult:
            if req.action == "open_positions":
                data = trading.get_open_positions()
                return ToolResult.success(data, f"{len(data.get('positions', []))} open positions")
            return ToolResult.error(f"Unsupported trading action: {req.action}")

        def analytics_handler(req: ToolRequest) -> ToolResult:
            if req.action == "summary":
                data = reporting.summary()
                return ToolResult.success(data, "Analytics summary generated")
            return ToolResult.error(f"Unsupported analytics action: {req.action}")

        registry.register("outreach", outreach_handler)
        registry.register("trading", trading_handler)
        registry.register("analytics", analytics_handler)

        inference = InferenceEngine(mode=mode)
        return cls(
            inference=inference,
            memory=SupabaseMemoryRepository(),
            emotion=EmotionModel(),
            identity=IdentityManager(),
            preferences=PreferenceEngine(),
            vitals=VitalsEngine(),
            router=ToolRouter(registry),
        )

    def _detect_tool_request(self, text: str) -> Optional[ToolRequest]:
        payload = (text or "").strip()
        if not payload:
            return None

        json_match = re.search(r"\{[\s\S]*\}", payload)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if {"tool_name", "action", "parameters"}.issubset(data.keys()):
                    return ToolRequest(
                        tool_name=str(data["tool_name"]),
                        action=str(data["action"]),
                        parameters=dict(data.get("parameters") or {}),
                    )
            except Exception:
                pass

        low = payload.lower()
        if any(k in low for k in ["prospect", "lead", "outreach"]):
            return ToolRequest("outreach", "generate_leads", {"product": "manifest", "limit": 5})
        if any(k in low for k in ["contact at", "find contact", "resolve contact"]):
            company = payload.split("contact")[-1].strip()
            return ToolRequest("outreach", "resolve_contact", {"company_name": company})
        if any(k in low for k in ["open positions", "trading positions"]):
            return ToolRequest("trading", "open_positions", {})
        if any(k in low for k in ["report", "analytics summary", "metrics summary"]):
            return ToolRequest("analytics", "summary", {})
        return None

    def _tool_result_to_text(self, result: ToolResult) -> str:
        if result.status != "success":
            return result.summary or "Tool request failed."
        data_preview = json.dumps(result.data, ensure_ascii=True)[:800]
        return f"{result.summary}. Data: {data_preview}"

    def think(
        self,
        user_input: str,
        *,
        identity: str = "sylana",
        active_tools: Optional[List[str]] = None,
        thread_id: Optional[int] = None,
        store: bool = True,
    ) -> Dict[str, Any]:
        self.turn_count += 1
        ident = self.identity.resolve(identity)
        emotion_vec = self.emotion.score(user_input)
        self.emotional_history.append(emotion_vec.category)

        self.vitals.update_from_user_event({"text": user_input, "mood": emotion_vec.valence})
        tool_req = self._detect_tool_request(user_input)

        if tool_req and (not active_tools or tool_req.tool_name in active_tools):
            tool_result = self.router.route(tool_req, context={"identity": ident.name})
            response_text = self._tool_result_to_text(tool_result)
            tool_payload = {
                "request": asdict(tool_req),
                "result": asdict(tool_result),
            }
        else:
            retrieval = self.memory.retrieve(
                RetrievalQuery(text=user_input, identity=ident.namespace, limit=8)
            )
            history = self.memory.history(ident.namespace, limit=5)
            system_prompt = ident.system_prompt
            prefs = self.preferences.get(ident.name)
            if prefs.get("tone"):
                system_prompt += f"\nPreferred tone: {prefs['tone']}."

            prompt = self.prompt_engineer.build_complete_prompt(
                system_message=system_prompt,
                user_input=user_input,
                emotion=emotion_vec.category,
                semantic_memories=retrieval.conversations,
                core_memories=retrieval.core_memories,
                recent_history=history,
                emotional_history=self.emotional_history[-5:],
            )
            response_text = self.inference.generate(
                system_prompt=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=320,
                active_tools=active_tools,
            ).strip() or "I'm here with you."
            tool_payload = None

        conversation_id = None
        if store:
            conversation_id = self.memory.store(
                MemoryRecord(
                    user_input=user_input,
                    sylana_response=response_text,
                    identity=ident.namespace,
                    memory_type=MemoryType.EMOTIONAL if emotion_vec.arousal > 0.65 else MemoryType.OPERATIONAL,
                    emotion=emotion_vec,
                    important=bool(tool_payload),
                    metadata={"thread_id": thread_id, "active_tools": active_tools or []},
                )
            )

        return {
            "response": response_text,
            "emotion": asdict(emotion_vec),
            "vitals": self.vitals.snapshot(),
            "tool": tool_payload,
            "identity": ident.name,
            "conversation_id": conversation_id,
            "turn": self.turn_count,
        }

    async def think_async(self, *args, **kwargs) -> Dict[str, Any]:
        return await asyncio.to_thread(self.think, *args, **kwargs)
