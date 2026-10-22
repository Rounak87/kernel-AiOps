"""
Autonomous Diagnostic Agent using Gemini native tool calling.

Orchestrates multi-turn investigations over kernel incidents, executing diagnostic
tools to verify or refute ML hypotheses and producing structured RCA reports.
"""

import os
import json
import logging
from dotenv import load_dotenv

from google import genai
from google.genai import types

from agent.tools import (
    TOOL_FUNCTIONS,
    query_metrics_history,
    get_service_topology,
    get_recent_deployments,
    search_logs,
)
from agent.prompts import SRE_SYSTEM_PROMPT, build_incident_brief

load_dotenv()
log = logging.getLogger(__name__)

_DEFAULT_MODEL = "gemini-3.5-flash-lite"


def _parse_rca(text: str) -> dict:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass
    return {"raw_report": text}


class DiagnosticAgent:
    """
    Autonomous SRE diagnostic agent powered by Gemini tool calling.
    """

    def __init__(self, api_key: str | None = None, model: str = _DEFAULT_MODEL):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set. Provide via .env or constructor.")
        self._client = genai.Client(api_key=key)
        self._model = model

    def investigate(self, incident, verbose: bool = True) -> dict:
        """
        Executes an autonomous investigation loop for a kernel incident.
        """
        incident_brief = build_incident_brief(incident)

        if verbose:
            print(f"\n{'='*70}")
            print(f"  AGENT INVESTIGATION: {incident.service} ({incident.status})")
            print(f"{'='*70}")
            print(f"\n[INCIDENT BRIEF]\n{incident_brief}")

        tools = [
            query_metrics_history,
            get_service_topology,
            get_recent_deployments,
            search_logs,
        ]

        config = types.GenerateContentConfig(
            system_instruction=SRE_SYSTEM_PROMPT,
            tools=tools,
            temperature=0.1,
        )

        chat = self._client.chats.create(model=self._model, config=config)
        response = chat.send_message(incident_brief)

        if verbose:
            print("[INVESTIGATION TRACE]")
            for msg in chat.get_history():
                for p in msg.parts:
                    if p.function_call:
                        print(f"  -> Tool Call: {p.function_call.name}({dict(p.function_call.args)})")
                    elif p.function_response:
                        preview = str(p.function_response.response)[:200]
                        print(f"  <- Observation: {preview}...")

        final_text = response.text or ""
        return _parse_rca(final_text)
