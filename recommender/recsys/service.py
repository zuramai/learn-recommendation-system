"""gRPC recommender service: reads the marketplace DB (read-only), recommends on request.

Rust calls this over gRPC (see proto/recommender.proto). Every Recommend call:
refresh the in-memory DB copy -> candidates -> features -> LightGBM -> re-rank,
with "now" as the cutoff, so a click logged a second ago already counts.
The ranker itself is retrained periodically (and on demand) on the same data.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent import futures
from datetime import datetime, timezone
from pathlib import Path

import grpc
import lightgbm as lgb
import pandas as pd

from . import pipeline, ranker, schema
from .db import Store
from .pb import recommender_pb2 as pb
from .pb import recommender_pb2_grpc as pb_grpc

log = logging.getLogger("recsys.service")
MODEL_DIR = Path(os.environ.get("RECSYS_MODEL_DIR", "artifacts"))


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def retrain(store: Store, model_dir: Path = MODEL_DIR) -> tuple[lgb.Booster, dict, dict]:
    """Train + evaluate on the current DB snapshot, save ranker.txt + model_meta.json.

    Returns (model, meta, report) where meta is what's written to model_meta.json.
    """
    trained_at = datetime.now(timezone.utc).replace(microsecond=0)
    model, report = pipeline.train_and_evaluate(store.snapshot(), end=_now())
    metrics = {
        pipeline.metric_key(method, metric): float(v)
        for method, row in report["results"].iterrows()
        for metric, v in row.items()
    }
    metrics["candidate_recall"] = float(report["candidate_recall"])
    meta = {
        "version": trained_at.strftime("%Y%m%d-%H%M%S"),
        "trained_at": trained_at.isoformat().replace("+00:00", "Z"),
        "metrics": metrics,
        "importance": {k: float(v) for k, v in report["importance"].items()},
        "train_rows": int(report["train_rows"]),
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    ranker.save(model, model_dir / "ranker.txt")
    (model_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))
    return model, meta, report


class RecommenderService(pb_grpc.RecommenderServicer):
    def __init__(self, store: Store, model_dir: Path = MODEL_DIR) -> None:
        self.store = store
        self.model_dir = model_dir
        self.retrain_lock = threading.Lock()
        store.refresh()
        if (model_dir / "ranker.txt").exists() and (model_dir / "model_meta.json").exists():
            self.model = ranker.load(model_dir / "ranker.txt")
            self.meta = json.loads((model_dir / "model_meta.json").read_text())
            log.info("loaded model %s", self.meta["version"])
        else:
            log.info("no model in %s/ — training one before serving", model_dir)
            self.retrain()

    def retrain(self) -> dict:
        """Raises RuntimeError if a retrain is already running."""
        if not self.retrain_lock.acquire(blocking=False):
            raise RuntimeError("retrain already running")
        try:
            self.store.refresh()
            t = time.perf_counter()
            model, meta, _ = retrain(self.store, self.model_dir)
            self.model, self.meta = model, meta  # swap: in-flight requests keep the old booster
            log.info("retrained model %s in %.1fs, ndcg@20=%.4f", meta["version"], time.perf_counter() - t,
                     meta["metrics"].get("lightgbm_ranker/ndcg@20", float("nan")))  # fmt: skip
            return meta
        finally:
            self.retrain_lock.release()

    # book:start recommend_rpc
    def Recommend(self, request, context):
        t = time.perf_counter()
        self.store.refresh()  # pick up events logged since the last request
        k = min(max(request.k, 1), 50)
        model, version = self.model, self.meta["version"]
        recs, n_candidates = pipeline.recommend(self.store.snapshot(), _now(), model, request.user_id, k)
        sources = pipeline.source_of(recs) if len(recs) else []
        contribs = ranker.contributions(model, recs) if len(recs) else []
        items = [
            pb.RecommendedItem(
                item_id=int(item_id), score=float(score), source=source, rank=rank,
                contributions=[pb.Contribution(feature=f, value=v, contribution=c) for f, v, c in contrib],
            )  # fmt: skip
            for rank, (item_id, score, source, contrib) in enumerate(
                zip(recs["item_id"], recs["score"], sources, contribs), start=1
            )
        ]
        latency_ms = (time.perf_counter() - t) * 1000
        log.info("recommend user=%d k=%d candidates=%d latency_ms=%.0f", request.user_id, k, n_candidates, latency_ms)
        return pb.RecommendResponse(
            items=items, model_version=version, candidates_considered=n_candidates, latency_ms=latency_ms
        )
    # book:end recommend_rpc

    def SimilarItems(self, request, context):
        t = time.perf_counter()
        self.store.refresh()
        k = min(max(request.k, 1), 30)
        sim = pipeline.similar_items(self.store.snapshot(), _now(), request.item_id, k)
        items = [
            pb.RecommendedItem(item_id=int(i), score=float(s), source=src, rank=rank)
            for rank, (i, s, src) in enumerate(zip(sim["item_id"], sim["score"], sim["source"]), start=1)
        ]
        return pb.RecommendResponse(
            items=items, model_version=self.meta["version"], candidates_considered=len(sim),
            latency_ms=(time.perf_counter() - t) * 1000,
        )  # fmt: skip

    def Retrain(self, request, context):
        try:
            meta = self.retrain()
        except RuntimeError as e:
            context.abort(grpc.StatusCode.ABORTED, str(e))
        return pb.RetrainResponse(model_version=meta["version"], metrics=meta["metrics"], train_rows=meta["train_rows"])

    def ModelInfo(self, request, context):
        meta = self.meta
        imp = meta.get("importance", {})
        feats = sorted(schema.FEATURES, key=lambda f: -imp.get(f.name, 0.0))
        return pb.ModelInfoResponse(
            model_version=meta["version"],
            trained_at=meta["trained_at"],
            metrics=meta["metrics"],
            features=[
                pb.FeatureInfo(name=f.name, group=f.group, description=f.description, gain_share=imp.get(f.name, 0.0))
                for f in feats
            ],
            events_loaded=self.store.events_loaded,
            train_rows=meta["train_rows"],
        )


def _retrain_loop(service: RecommenderService, interval: int) -> None:
    while True:
        time.sleep(interval)
        try:
            service.retrain()
        except RuntimeError:
            log.info("periodic retrain skipped: one is already running")
        except Exception:
            log.exception("periodic retrain failed")


def serve() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    addr = os.environ.get("RECSYS_GRPC_ADDR", "127.0.0.1:50051")
    service = RecommenderService(Store())
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_RecommenderServicer_to_server(service, server)
    server.add_insecure_port(addr)
    server.start()
    interval = int(os.environ.get("RECSYS_RETRAIN_INTERVAL_SEC", "21600"))
    if interval > 0:
        threading.Thread(target=_retrain_loop, args=(service, interval), daemon=True).start()
    log.info("listening on %s (model %s, %d events loaded)", addr, service.meta["version"], service.store.events_loaded)
    server.wait_for_termination()
