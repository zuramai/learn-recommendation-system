"""Chapter 13 - end-to-end latency of the live gRPC recommender (Recommend, k=20, 200 random users)."""

import sys
from pathlib import Path

import grpc
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402
from recsys.pb import recommender_pb2, recommender_pb2_grpc  # noqa: E402

ADDR = "127.0.0.1:50051"
N_USERS = 200


def main() -> None:
    channel = grpc.insecure_channel(ADDR)
    stub = recommender_pb2_grpc.RecommenderStub(channel)
    try:
        grpc.channel_ready_future(channel).result(timeout=3)
    except grpc.FutureTimeoutError:
        print("start it: cd recommender && uv run recsys serve")
        sys.exit(1)

    rng = np.random.default_rng(13)
    user_ids = rng.choice(np.arange(1, 1501), size=N_USERS, replace=False)

    common.section(f"Recommend(k=20) for {N_USERS} random users at {ADDR}")
    try:
        warm = stub.Recommend(recommender_pb2.RecommendRequest(user_id=int(user_ids[0]), k=20, surface="home_feed"), timeout=60)
    except grpc.RpcError as e:
        print(f"RPC failed: {e.code()} {e.details()}\nstart it: cd recommender && uv run recsys serve")
        sys.exit(1)
    print(f"model_version={warm.model_version} (first call is a warm-up and not counted)")

    server_ms, wall_ms, cands, n_items = [], [], [], []
    import time

    for uid in user_ids:
        t0 = time.perf_counter()
        r = stub.Recommend(recommender_pb2.RecommendRequest(user_id=int(uid), k=20, surface="home_feed"), timeout=60)
        wall_ms.append((time.perf_counter() - t0) * 1000)
        server_ms.append(r.latency_ms)
        cands.append(r.candidates_considered)
        n_items.append(len(r.items))

    def pct(a):
        return {f"p{p}": float(np.percentile(a, p)) for p in (50, 90, 99)} | {"mean": float(np.mean(a)), "max": float(np.max(a))}

    srv, wall = pct(server_ms), pct(wall_ms)
    print(f"{'':<22}{'p50':>9}{'p90':>9}{'p99':>9}{'mean':>9}{'max':>9}")
    for label, d in [("server latency_ms", srv), ("client wall ms", wall)]:
        print(f"{label:<22}" + "".join(f"{d[k]:>9.1f}" for k in ("p50", "p90", "p99", "mean", "max")))
    print(f"\ncandidates scored per request: mean {np.mean(cands):.0f}, min {min(cands)}, max {max(cands)}")
    print(f"items returned per request:   mean {np.mean(n_items):.1f}")

    common.save_results("ch13_latency", {
        "n_users": N_USERS, "model_version": warm.model_version, "server_ms": server_ms, "wall_ms": wall_ms,
        "server": srv, "wall": wall, "candidates_mean": float(np.mean(cands)),
    })


if __name__ == "__main__":
    main()
