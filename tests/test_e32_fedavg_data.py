from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from flask import request as flask_request

from exp_manager.services.e32_data import build_paired_assignments
from server.app import create_app


class LabelLoader:
    def labels(self, path):
        if path.name == "train":
            return [0] * 11 + [1] * 14 + [2] * 15 + [0, 1, 2]
        return [0, 1]

    def record_count(self, path):
        return len(self.labels(path))


def manifest(tmp_path):
    return {
        "scope": "smoke",
        "source_dir": str(tmp_path),
        "clients": [
            {"client_id": client, "ranges": [{"file": "train", "start": 10 * client, "end": 10 * (client + 1)}]}
            for client in range(4)
        ],
        "test_ranges": [{"file": "train", "start": 40, "end": 43}],
        "final_test_ranges": [{"file": "test", "start": 0, "end": 2}],
    }


@pytest.mark.parametrize("sample_limit,per_client", [(None, 10), (5, 5)])
def test_uneven_classes_still_assign_the_same_pool_and_equal_sample_counts(tmp_path, sample_limit, per_client):
    source = manifest(tmp_path)
    options = {"training_samples_per_client": sample_limit}
    paired = build_paired_assignments(source, 101, LabelLoader(), **options)
    assert paired == build_paired_assignments(source, 101, LabelLoader(), **options)
    assert paired["validation_ranges"] == source["test_ranges"]
    assert paired["final_test_ranges"] == source["final_test_ranges"]
    assigned_pools = []
    for clients, counts in paired["conditions"].values():
        assert [row["num_samples"] for row in counts] == [per_client] * 4
        assigned = [index for client in clients for row in client["indices_by_file"] for index in row["indices"]]
        assert len(assigned) == len(set(assigned)) == per_client * 4
        assert set(assigned) <= set(range(40))
        assigned_pools.append(set(assigned))
    assert assigned_pools[0] == assigned_pools[1]
    assert paired["conditions"]["iid"][0] != paired["conditions"]["non_iid"][0]
    iid, skew = (paired["conditions"][name][1] for name in ("iid", "non_iid"))
    assert all(max(counts) - min(counts) <= 1 for counts in zip(*(row["class_counts"].values() for row in iid)))
    assert sum(max(row["class_counts"].values()) for row in skew) > sum(max(row["class_counts"].values()) for row in iid)


@pytest.mark.parametrize("split,start,end", [("test_ranges", 39, 41), ("final_test_ranges", 41, 43)])
def test_overlapping_index_is_rejected_even_with_an_alias(tmp_path, split, start, end):
    source = manifest(tmp_path)
    source[split] = [{"file": "./train", "start": start, "end": end}]
    with pytest.raises(ValueError, match="indices overlap"):
        build_paired_assignments(source, 101, LabelLoader())


def test_final_test_uses_completed_model_and_caches_evaluation(monkeypatch):
    app = create_app("127.0.0.1", 5000)
    model = {"experiment_id": "normal", "model_version": 2}
    validation = {"source_dir": "fixture", "ranges": [{"file": "train", "start": 40, "end": 43}], "split": "validation"}
    final_test = {"source_dir": "fixture", "ranges": [{"file": "test", "start": 0, "end": 2}], "split": "test"}
    calls = []

    def evaluate(actual_model, dataset, config, round_id):
        calls.append((actual_model, dataset, config, round_id))
        sample_range, = dataset["ranges"]
        count = sample_range["end"] - sample_range["start"]
        return {"split": "test", "num_test_samples": count, "correct_predictions": 1, "accuracy": 1 / count}

    monkeypatch.setattr("server.routes.experiment.evaluate_global_model", evaluate)
    with app.test_client() as client:
        dataset_response = client.post("/api/dataset/range", json={**validation, "final_test_range": final_test})
        assert dataset_response.status_code == 200
        assert client.get("/api/dataset/range").get_json() == validation
        app.extensions.update(global_model=model, round_config={"final_test_round": 2, "max_test_samples": 1})
        request = {"experiment_id": "normal", "round": 2}
        assert client.post("/api/evaluation/final-test", json=request).status_code == 409
        app.extensions.update(
            completed_rounds={1, 2},
            round_results=[{"round": 2, "global_model": {"checkpoint": "global_0002.pt"}}],
        )
        first = client.post("/api/evaluation/final-test", json=request)
        assert client.post("/api/dataset/range", json={**validation, "final_test_range": final_test}).status_code == 200
        again = client.post("/api/evaluation/final-test", json=request)
        assert first.status_code == again.status_code == 200
        assert first.get_json() == again.get_json()
        assert first.get_json()["model_version"] == 2
        assert calls == [(model, final_test, {"final_test_round": 2, "max_test_samples": None}, 2)]

        replacement = {**final_test, "ranges": [{"file": "test", "start": 2, "end": 5}]}
        assert client.post("/api/dataset/range", json={**validation, "final_test_range": replacement}).status_code == 200
        reevaluated = client.post("/api/evaluation/final-test", json=request)
        assert reevaluated.status_code == 200
        assert reevaluated.get_json()["num_test_samples"] == 3
        assert [call[1] for call in calls] == [final_test, replacement]

        assert client.post("/api/dataset/range", json=validation).status_code == 200
        assert app.extensions["final_test_range"] is None
        assert app.extensions["final_test_evaluation"] is None
        unavailable = client.post("/api/evaluation/final-test", json=request)
        assert unavailable.status_code == 409
        assert unavailable.get_json()["error"] == "Distribute the official final test dataset first"


def test_replacing_final_test_range_invalidates_in_flight_evaluation(monkeypatch):
    app = create_app("127.0.0.1", 5000)
    final_test = {"source_dir": "fixture", "ranges": [{"file": "test", "start": 0, "end": 2}], "split": "test"}
    replacement = {**final_test, "ranges": [{"file": "test", "start": 2, "end": 5}]}
    app.extensions.update(
        final_test_range=final_test,
        global_model={"experiment_id": "normal", "model_version": 1},
        round_config={"final_test_round": 1},
        completed_rounds={1},
        round_results=[{"round": 1, "global_model": {"checkpoint": "global_0001.pt"}}],
    )
    evaluate_started = Event()
    replace_started = Event()
    finish_evaluation = Event()
    lock = app.extensions["final_test_lock"]

    class ObservedLock:
        def __enter__(self):
            if flask_request.path == "/api/dataset/range":
                replace_started.set()
            lock.acquire()

        def __exit__(self, *_):
            lock.release()

    def evaluate(_model, dataset, _config, _round_id):
        if dataset == final_test:
            evaluate_started.set()
            assert finish_evaluation.wait(timeout=5)
        return {"split": "test", "evaluated_ranges": dataset["ranges"]}

    def post(path, payload):
        with app.test_client() as client:
            return client.post(path, json=payload)

    app.extensions["final_test_lock"] = ObservedLock()
    monkeypatch.setattr("server.routes.experiment.evaluate_global_model", evaluate)
    evaluation_path = "/api/evaluation/final-test"
    evaluation_request = {"experiment_id": "normal", "round": 1}
    with ThreadPoolExecutor(max_workers=2) as workers:
        evaluation = workers.submit(post, evaluation_path, evaluation_request)
        try:
            assert evaluate_started.wait(timeout=5)
            change = workers.submit(post, "/api/dataset/range", {
                "source_dir": "fixture", "split": "validation",
                "ranges": [{"file": "train", "start": 40, "end": 43}],
                "final_test_range": replacement,
            })
            assert replace_started.wait(timeout=5)
            assert app.extensions["final_test_range"] == final_test
        finally:
            finish_evaluation.set()

        assert evaluation.result(timeout=5).get_json()["evaluated_ranges"] == final_test["ranges"]
        assert change.result(timeout=5).status_code == 200

    assert app.extensions["final_test_evaluation"] is None
    assert post(evaluation_path, evaluation_request).get_json()["evaluated_ranges"] == replacement["ranges"]
