"""
Smoke test for the Granger Causality Engine against live InfluxDB telemetry.
Run with: python -m models.test_causal
"""
import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

from models.causal_engine import CausalEngine

def main():
    engine = CausalEngine()

    test_targets = ["multipathd", "jbd2/nvme0n1p1-", "ml-engine"]

    print("=" * 70)
    print("       GRANGER CAUSALITY ROOT CAUSE ANALYSIS ENGINE")
    print("=" * 70)

    for svc in test_targets:
        print(f"\nEvaluating target service: [{svc}]")
        print("-" * 50)
        res = engine.analyze(svc)
        print(f"Active telemetry metrics: {res.active_metrics}")
        print(f"Root Cause Metric:        {res.root_cause or 'None'}")
        print(f"Summary:                  {res.summary}")
        
        if res.edges:
            print(f"\nDetected Causal Edges ({len(res.edges)}):")
            for edge in res.edges:
                print(f"  {edge.source:<20} --(lag={edge.best_lag}s, conf={edge.confidence*100:.1f}%)--> {edge.target:<20}")
        else:
            print("No directed causal edges passed p < 0.05 threshold.")
        print("=" * 70)

if __name__ == "__main__":
    main()
