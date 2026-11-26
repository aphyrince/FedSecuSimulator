import json
from unittest.mock import Mock

import pytest

from experiments.run_e32_sequential_smoke import run_comparison


def _completed_result(
    directory, count, *, model_hash="initial", excluded=(), exclusion_reason="virtual_deadline_exceeded",
    distribution="iid", image_hash="images", assignment="iid",
):
    directory.mkdir()
    (directory / "experiment_info.json").write_text(
        json.dumps({"data_split": {
            "split_id": f"e32-{distribution}", "client_ranges": [assignment],
            "training_image_set": {"sha256": image_hash, "sample_count": 160},
        }}), encoding="utf-8",
    )
    result_path = directory / "results.json"
    result_path.write_text(json.dumps({
        "config": {"seed": 101},
        "round_results": [{
            "model_updated": True, "accepted_count": count, "accepted_samples": 16 * count,
            "trained_count": count + len(excluded), "excluded_clients": list(excluded),
            "client_activity": [{"client_id": client_id, "reason": exclusion_reason} for client_id in excluded],
            "client_execution_log": str(directory / "client_execution" / "round_001.json"),
            "start_model_hash": model_hash,
            "evaluation": {"dataset_fingerprint": "validation", "num_test_samples": 64},
        }],
        "final_test": {"dataset_fingerprint": "official-test", "num_test_samples": 10000},
    }), encoding="utf-8")
    return {"status": "completed", "result_path": str(result_path)}


@pytest.mark.parametrize(
    "mismatch,error,started_count",
    [
        (None, None, 4),
        ("model", "different model, seed or data split", 2),
        ("delay", "unexpected training or virtual-deadline exclusions", 3),
        ("pool", "different model, seed or data split", 4),
        ("assignment", "unexpected training assignment", 4),
    ],
)
def test_e32_smoke_waits_for_each_case_and_checks_shared_inputs(monkeypatch, tmp_path, mismatch, error, started_count):
    manager = Mock()
    data_manager = Mock()
    manager.update_config.return_value = {"round_timeout": 10}
    manager.status.side_effect = [
        {"status": "running"}, _completed_result(tmp_path / "full", 10),
        {"status": "running"}, _completed_result(tmp_path / "half", 5, model_hash="different" if mismatch == "model" else "initial"),
        {"status": "running"}, _completed_result(
            tmp_path / "delay", 7, excluded=(2, 5, 8),
            exclusion_reason="real_submission_deadline_exceeded" if mismatch == "delay" else "virtual_deadline_exceeded",
        ),
        {"status": "running"}, _completed_result(
            tmp_path / "label-bias", 10, distribution="non_iid", image_hash="different" if mismatch == "pool" else "images",
            assignment="iid" if mismatch == "assignment" else "non_iid",
        ),
    ]
    monkeypatch.setattr("experiments.run_e32_sequential_smoke.time.sleep", lambda _: None)

    if error is None:
        results = run_comparison(manager, data_manager, "cpu")
        assert [item["accepted_count"] for item in results] == [10, 5, 7, 10]
        assert [item["trained_count"] for item in results] == [10, 5, 10, 10]
        assert results[2]["excluded_clients"] == [2, 5, 8]
        assert results[3]["split_id"] != results[0]["split_id"]
    else:
        with pytest.raises(RuntimeError, match=error):
            run_comparison(manager, data_manager, "cpu")

    data_manager.distribute.assert_called_once_with(seed=101, distribution="iid", data_protocol="e32")
    assert [call[0] for call in manager.mock_calls] == ["update_config", "start", "status", "status"] * started_count
    configs = [call.args[0] for call in manager.update_config.call_args_list]
    first = configs[0]
    for config, (suffix, rate, delays, distribution) in zip(configs, (
        ("full", 1.0, [], "iid"), ("half", 0.5, [], "iid"),
        ("delay", 1.0, [2, 5, 8], "iid"), ("label-bias", 1.0, [], "non_iid"),
    )):
        assert config["experiment_id"].endswith(f"-{suffix}")
        assert config == {
            **first, "experiment_id": config["experiment_id"], "participation_rate": rate,
            "delay_clients": delays, "data_distribution": distribution,
        }
    assert first["model_config"]["architecture"] == "resnet18_cifar10_v1"
    assert first["num_rounds"] == 1 and first["max_train_samples"] == 16
    assert first["delay_clients"] == []
    assert not first["attack_enabled"] and not first["defense_enabled"]


def test_e32_smoke_does_not_start_the_second_case_after_failure():
    manager = Mock()
    manager.update_config.return_value = {"round_timeout": 10}
    manager.status.return_value = {"status": "failed", "message": "first failed"}

    with pytest.raises(RuntimeError, match="first failed"):
        run_comparison(manager, Mock(), "cpu")

    manager.update_config.assert_called_once()
    manager.start.assert_called_once()
