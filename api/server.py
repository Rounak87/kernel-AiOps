"""
FastAPI Server and Autonomous Incident Hub.
Provides REST endpoints and an interactive web console connecting
the eBPF telemetry, tri-model ML engine, and the Gemini SRE Agent.
"""

import os
import time
import logging
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from dotenv import load_dotenv

from models.engine import MLAnalyticsEngine, IncidentDiagnosis
from agent.core import DiagnosticAgent
from pipeline.db import list_active_services, fetch_latest

load_dotenv()
log = logging.getLogger("server")

app = FastAPI(
    title="Kernel AIOps — Autonomous SRE Platform",
    description="eBPF Kernel Telemetry, Tri-Model ML Anomaly & Causal Engine, and Gemini SRE Agent",
    version="1.0.0",
)

# Global engine instances
ml_engine = MLAnalyticsEngine()
agent = DiagnosticAgent()

# Incident cache for recent investigations
INCIDENT_STORE: List[Dict[str, Any]] = []
_SERVICES_CACHE: Dict[str, Any] = {"timestamp": 0, "data": []}


@app.get("/api/v1/health")
def health_check():
    gemini_ready = bool(os.environ.get("GEMINI_API_KEY"))
    active_services = list_active_services(window="30d")
    return {
        "status": "healthy",
        "influxdb": "connected",
        "active_services_count": len(active_services),
        "gemini_agent": "ready" if gemini_ready else "missing_key",
        "grafana_url": "http://localhost:3000/d/kernel-aiops-telemetry/kernel-aiops-telemetry",
    }


@app.get("/api/v1/services")
def get_services():
    now = time.time()
    if now - _SERVICES_CACHE["timestamp"] < 30 and _SERVICES_CACHE["data"]:
        return _SERVICES_CACHE["data"]

    # Target high-interest services first, plus a sample of active background services
    priority_targets = ["multipathd", "ml-engine", "jbd2/nvme0n1p1-", "sshd", "eic_parse_autho"]
    all_active = list_active_services(window="30d")
    for svc in all_active:
        if svc not in priority_targets and len(priority_targets) < 12:
            priority_targets.append(svc)

    results = []
    for svc in priority_targets:
        diag = ml_engine.diagnose(svc)
        snapshot = fetch_latest(svc, window="30d") or {}
        results.append({
            "service": svc,
            "status": diag.status,
            "anomaly_score": round(diag.anomaly_score, 4),
            "is_anomalous": diag.is_anomalous,
            "root_cause_metric": diag.root_cause_metric or "N/A",
            "time_to_saturation": diag.time_to_saturation_seconds,
            "is_saturated": diag.is_saturated,
            "causal_edges": diag.causal_edges_count,
            "avg_stall_ms": round((snapshot.get("avg_stall_ns") or 0) / 1e6, 2),
            "avg_lock_ms": round((snapshot.get("avg_lock_wait_ns") or 0) / 1e6, 2),
            "major_faults": snapshot.get("major_faults", 0),
        })

    results.sort(key=lambda x: (x["status"] != "CRITICAL", x["status"] != "WARNING", -x["anomaly_score"]))
    _SERVICES_CACHE["timestamp"] = now
    _SERVICES_CACHE["data"] = results
    return results


@app.get("/api/v1/diagnose/{service}")
def diagnose_service(service: str):
    diag = ml_engine.diagnose(service)
    snapshot = fetch_latest(service, window="30d") or {}
    return {
        "service": diag.service,
        "timestamp": diag.timestamp,
        "status": diag.status,
        "anomaly_score": diag.anomaly_score,
        "is_anomalous": diag.is_anomalous,
        "root_cause_metric": diag.root_cause_metric,
        "time_to_saturation_seconds": diag.time_to_saturation_seconds,
        "is_saturated": diag.is_saturated,
        "causal_summary": diag.causal_summary,
        "forecast_summary": diag.forecast_summary,
        "metrics_snapshot": snapshot,
    }


@app.post("/api/v1/investigate/{service}")
def investigate_service(service: str):
    diag = ml_engine.diagnose(service)
    rca = agent.investigate(diag, verbose=False)
    entry = {
        "service": service,
        "timestamp": diag.timestamp,
        "status": diag.status,
        "rca": rca,
    }
    INCIDENT_STORE.insert(0, entry)
    return entry


@app.get("/api/v1/incidents")
def list_incidents():
    return INCIDENT_STORE[:20]


@app.get("/", response_class=HTMLResponse)
def index_page():
    return HTML_DASHBOARD


HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Kernel AIOps — Autonomous SRE Command Center</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #111726;
      --card-border: #1e293b;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --primary: #6366f1;
      --primary-hover: #4f46e5;
      --danger: #ef4444;
      --danger-bg: rgba(239, 68, 68, 0.12);
      --warning: #f59e0b;
      --warning-bg: rgba(245, 158, 11, 0.12);
      --success: #10b981;
      --success-bg: rgba(16, 185, 129, 0.12);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
    }
    .container { max-width: 1400px; margin: 0 auto; }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 24px;
      border-bottom: 1px solid var(--card-border);
      margin-bottom: 24px;
    }
    .logo-group h1 { font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em; }
    .logo-group p { font-size: 0.875rem; color: var(--text-muted); }
    .status-badges { display: flex; gap: 12px; align-items: center; }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 12px;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      border: 1px solid var(--card-border);
      background: var(--card-bg);
    }
    .badge.green .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--success); }
    .badge.purple .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--primary); }
    .btn-grafana {
      background: #ff7800;
      color: #fff;
      text-decoration: none;
      padding: 8px 16px;
      border-radius: 8px;
      font-size: 0.8125rem;
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
    }
    .btn-grafana:hover { opacity: 0.9; transform: translateY(-1px); }

    .grid-layout {
      display: grid;
      grid-template-columns: 7fr 5fr;
      gap: 24px;
    }
    @media (max-width: 1024px) { .grid-layout { grid-template-columns: 1fr; } }

    .panel {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .panel-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }
    .panel-title { font-size: 1.1rem; font-weight: 700; }
    .panel-subtitle { font-size: 0.8rem; color: var(--text-muted); }

    table { width: 100%; border-collapse: collapse; text-align: left; font-size: 0.875rem; }
    th { color: var(--text-muted); font-weight: 600; padding: 12px 14px; border-bottom: 1px solid var(--card-border); }
    td { padding: 14px; border-bottom: 1px solid rgba(255,255,255,0.05); }
    tr:hover td { background: rgba(255,255,255,0.02); }

    .tag-critical { background: var(--danger-bg); color: var(--danger); padding: 4px 8px; border-radius: 6px; font-weight: 700; font-size: 0.75rem; border: 1px solid rgba(239, 68, 68, 0.3); }
    .tag-warning  { background: var(--warning-bg); color: var(--warning); padding: 4px 8px; border-radius: 6px; font-weight: 700; font-size: 0.75rem; border: 1px solid rgba(245, 158, 11, 0.3); }
    .tag-healthy  { background: var(--success-bg); color: var(--success); padding: 4px 8px; border-radius: 6px; font-weight: 700; font-size: 0.75rem; border: 1px solid rgba(16, 185, 129, 0.3); }

    .btn-action {
      background: var(--primary);
      color: white;
      border: none;
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }
    .btn-action:hover { background: var(--primary-hover); }

    .terminal-box {
      background: #050811;
      border: 1px solid #1e293b;
      border-radius: 8px;
      padding: 16px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.8rem;
      min-height: 280px;
      max-height: 400px;
      overflow-y: auto;
      color: #cbd5e1;
      white-space: pre-wrap;
    }
    .rca-card {
      margin-top: 16px;
      background: rgba(99, 102, 241, 0.05);
      border: 1px solid rgba(99, 102, 241, 0.2);
      border-radius: 8px;
      padding: 16px;
      display: none;
    }
    .rca-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
    .rca-badge { background: var(--primary); color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }
    .evidence-item { margin-bottom: 6px; padding-left: 12px; border-left: 2px solid var(--primary); font-size: 0.8125rem; }
    .remedy-btn { background: #334155; color: #fff; border: 1px solid #475569; padding: 6px 12px; border-radius: 6px; font-size: 0.75rem; margin-right: 8px; margin-top: 8px; cursor: pointer; }
    .remedy-btn:hover { background: #475569; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="logo-group">
        <h1>Kernel AIOps — Autonomous SRE Command Center</h1>
        <p>eBPF Kernel Tracepoints &bull; Tri-Model ML Analytics &bull; Autonomous Gemini Diagnostic Agent</p>
      </div>
      <div class="status-badges">
        <div class="badge green"><span class="dot"></span> InfluxDB Live</div>
        <div class="badge purple"><span class="dot"></span> Gemini Agent Active</div>
        <a href="http://localhost:3000/d/kernel-aiops-telemetry/kernel-aiops-telemetry" target="_blank" class="btn-grafana">
          <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 14.5v-9l6 4.5-6 4.5z"/></svg>
          Open Grafana Cockpit
        </a>
      </div>
    </header>

    <div class="grid-layout">
      <!-- Left Column: Fleet Health Monitor -->
      <div class="panel">
        <div class="panel-header">
          <div>
            <div class="panel-title">Fleet Telemetry &amp; Anomaly Status</div>
            <div class="panel-subtitle">Evaluated continuously via Isolation Forest + Granger DAG Engine</div>
          </div>
          <button class="btn-action" onclick="loadServices()">Refresh Fleet</button>
        </div>

        <table>
          <thead>
            <tr>
              <th>Service</th>
              <th>Status</th>
              <th>Anomaly Score</th>
              <th>Root Cause Metric</th>
              <th>Stall (ms)</th>
              <th>Lock Wait (ms)</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody id="services-table">
            <tr><td colspan="7" style="text-align: center; color: var(--text-muted);">Loading live service telemetry...</td></tr>
          </tbody>
        </table>
      </div>

      <!-- Right Column: Autonomous AI Incident Investigator -->
      <div class="panel">
        <div class="panel-header">
          <div>
            <div class="panel-title">Autonomous Gemini SRE Agent</div>
            <div class="panel-subtitle">Multi-Turn ReAct Reasoning &amp; Tool Calling</div>
          </div>
        </div>

        <div id="terminal" class="terminal-box">Click "Investigate" on any degraded service to launch the autonomous agent.

The agent will execute live diagnostic tools:
  [1] query_metrics_history() -> InfluxDB
  [2] get_service_topology()  -> Dependency DAG
  [3] get_recent_deployments() -> Commit audit
  [4] search_logs()           -> Kernel call traces</div>

        <div id="rca-card" class="rca-card">
          <div class="rca-header">
            <h3 id="rca-title" style="font-size: 1rem; font-weight: 700;">Root Cause Analysis</h3>
            <span id="rca-confidence" class="rca-badge">HIGH CONFIDENCE</span>
          </div>
          <p id="rca-summary" style="font-size: 0.85rem; margin-bottom: 12px; color: #e2e8f0;"></p>
          <div style="font-weight: 700; font-size: 0.8rem; color: var(--text-muted); margin-bottom: 6px;">CORROBORATED EVIDENCE:</div>
          <div id="rca-evidence"></div>
          <div style="font-weight: 700; font-size: 0.8rem; color: var(--text-muted); margin-top: 12px; margin-bottom: 6px;">RECOMMENDED REMEDIATION:</div>
          <div id="rca-remediation"></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    async function loadServices() {
      try {
        const res = await fetch('/api/v1/services');
        const data = await res.json();
        const tbody = document.getElementById('services-table');
        tbody.innerHTML = '';

        data.forEach(s => {
          const tr = document.createElement('tr');
          const tagClass = s.status === 'CRITICAL' ? 'tag-critical' : (s.status === 'WARNING' ? 'tag-warning' : 'tag-healthy');
          tr.innerHTML = `
            <td style="font-weight: 600;">${s.service}</td>
            <td><span class="${tagClass}">${s.status}</span></td>
            <td>${s.anomaly_score}</td>
            <td><code style="color: #a5b4fc;">${s.root_cause_metric}</code></td>
            <td>${s.avg_stall_ms} ms</td>
            <td>${s.avg_lock_ms} ms</td>
            <td>
              <button class="btn-action" onclick="runInvestigation('${s.service}')">Investigate</button>
            </td>
          `;
          tbody.appendChild(tr);
        });
      } catch (err) {
        console.error('Failed to load services:', err);
      }
    }

    async function runInvestigation(service) {
      const term = document.getElementById('terminal');
      const rcaCard = document.getElementById('rca-card');
      rcaCard.style.display = 'none';

      term.innerText = `[AIOPS ORCHESTRATOR] Initializing Autonomous Investigation for service: ${service}\\n`;
      term.innerText += `[ML ENGINES] Isolation Forest + Granger DAG active.\\n`;
      term.innerText += `[GEMINI SRE AGENT] Persona initialized. Generating incident brief...\\n`;
      term.innerText += `[AGENT REASONING] Contacting Gemini for multi-turn tool calling loop...\\n\\n`;

      try {
        const res = await fetch(`/api/v1/investigate/${service}`, { method: 'POST' });
        const result = await res.json();
        const rca = result.rca;

        term.innerText += `[AGENT COMPLETE] Root Cause identified: ${rca.root_cause_metric || 'N/A'}\\n`;
        term.innerText += `[CONFIDENCE] ${rca.confidence || 'HIGH'}\\n`;
        term.innerText += `[CAUSAL CHAIN] ${rca.causal_chain || 'N/A'}\\n`;
        term.innerText += `[DEPLOYMENT] ${rca.deployment_correlation || 'N/A'}\\n`;
        term.scrollTop = term.scrollHeight;

        // Render formatted RCA card
        document.getElementById('rca-title').innerText = `RCA: ${rca.root_cause_service || service} (${rca.root_cause_metric || ''})`;
        document.getElementById('rca-confidence').innerText = `${rca.confidence || 'HIGH'} CONFIDENCE`;
        document.getElementById('rca-summary').innerText = rca.summary || 'Investigation completed.';

        const evDiv = document.getElementById('rca-evidence');
        evDiv.innerHTML = '';
        (rca.evidence || []).forEach(ev => {
          const d = document.createElement('div');
          d.className = 'evidence-item';
          d.innerText = ev;
          evDiv.appendChild(d);
        });

        const remDiv = document.getElementById('rca-remediation');
        remDiv.innerHTML = '';
        (rca.remediation || []).forEach(rem => {
          const b = document.createElement('button');
          b.className = 'remedy-btn';
          b.innerText = rem;
          remDiv.appendChild(b);
        });

        rcaCard.style.display = 'block';
      } catch (err) {
        term.innerText += `\\n[ERROR] Investigation failed: ${err.message}`;
      }
    }

    // Auto-load on startup
    loadServices();
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)
