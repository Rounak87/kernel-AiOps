"""
End-to-end agent investigation runner for targeted services.
Run with: python -m agent.test_agent
"""
import time
import json
import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

from models.engine import MLAnalyticsEngine
from agent.core import DiagnosticAgent


def main():
    target_services = ["multipathd", "ml-engine"]
    print(f"Executing Agent Investigation for targets: {target_services}")

    ml_engine = MLAnalyticsEngine()
    agent = DiagnosticAgent()

    for idx, svc in enumerate(target_services):
        incident = ml_engine.diagnose(svc)
        rca = agent.investigate(incident, verbose=True)

        print(f"\n{'='*70}")
        print(f"  STRUCTURED RCA REPORT: {svc.upper()}")
        print(f"{'='*70}")
        print(json.dumps(rca, indent=2))
        print()

        # Respect Gemini free-tier RPM quota between investigations
        if idx < len(target_services) - 1:
            print("Pacing API requests (pausing 15s before next investigation)...")
            time.sleep(15)


if __name__ == "__main__":
    main()
