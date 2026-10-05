from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging
import time
from pipeline.db import write_metric_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("ingest")

app = FastAPI(title="kernel-aiops ingest", docs_url="/docs")


class MetricBatch(BaseModel):
    node: str
    timestamp: float
    metrics: list[dict]


@app.get("/health")
def health():
    return {"status": "ok", "time": time.time()}


@app.post("/api/v1/ingest")
def ingest(batch: MetricBatch):
    if not batch.metrics:
        raise HTTPException(status_code=400, detail="empty metrics list")

    written = write_metric_batch(node=batch.node, metrics=batch.metrics)
    log.info(f"node={batch.node}  records={written}")

    return {"written": written}


if __name__ == "__main__":
    uvicorn.run("pipeline.ingest:app", host="0.0.0.0", port=5000, reload=False)
