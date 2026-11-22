import pytest

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


def test_uneven_classes_still_assign_the_same_pool_and_equal_sample_counts(tmp_path):
    source = manifest(tmp_path)
    paired = build_paired_assignments(source, 101, LabelLoader())
    assert paired == build_paired_assignments(source, 101, LabelLoader())
    for clients, counts in paired["conditions"].values():
        assert [row["num_samples"] for row in counts] == [10] * 4
        assigned = [index for client in clients for row in client["indices_by_file"] for index in row["indices"]]
        assert sorted(assigned) == list(range(40))
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
        return {"split": "test", "num_test_samples": 2, "correct_predictions": 1, "accuracy": 0.5, "dataset_fingerprint": "test-hash"}

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
        again = client.post("/api/evaluation/final-test", json=request)
        assert first.status_code == again.status_code == 200
        assert first.get_json() == again.get_json()
        assert first.get_json()["model_version"] == 2
        assert calls == [(model, final_test, {"final_test_round": 2, "max_test_samples": None}, 2)]
        client.post("/api/dataset/range", json=validation)
        assert app.extensions["final_test_range"] is None
