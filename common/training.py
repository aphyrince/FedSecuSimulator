"""공통 학습 설정과 순차 FedAvg 라운드 일정."""

import random

from fedsec.config import TrainingSettings, setup_runtime
from fedsec.validation import finite_number, positive_int


DEFAULT_VIRTUAL_TIMING = {
    "normal_arrival_seconds": 1.0,
    "delayed_arrival_seconds": 8.0,
    "deadline_seconds": 5.0,
}


def validate_sequential_round(config: dict) -> None:
    if (
        config.get("round_type", "federated") != "federated"
        or config.get("attack_enabled")
        or config.get("defense_enabled", config.get("defense", "none") != "none")
    ):
        raise ValueError("순차 실행은 정상 FedAvg를 지원합니다.")
    if config.get("participation_rate") not in (0.5, 1.0):
        raise ValueError("순차 실행의 참여율은 0.5 또는 1.0이어야 합니다.")
    if config.get("delay_clients") and config["participation_rate"] != 1.0:
        raise ValueError("가상 지연 조건에서는 모든 클라이언트가 학습합니다.")


def virtual_timing(value: dict | None = None) -> dict:
    if value is None:
        return dict(DEFAULT_VIRTUAL_TIMING)
    if not isinstance(value, dict) or value.keys() - DEFAULT_VIRTUAL_TIMING.keys():
        raise ValueError("잘못된 가상 시간 설정입니다.")
    result = {**DEFAULT_VIRTUAL_TIMING, **value}
    for name, seconds in result.items():
        if finite_number(seconds, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    if result["normal_arrival_seconds"] > result["deadline_seconds"]:
        raise ValueError("정상 도착 시간은 가상 마감 이하여야 합니다.")
    if result["delayed_arrival_seconds"] < result["normal_arrival_seconds"]:
        raise ValueError("지연 대상의 도착 시간은 정상 도착 시간 이상이어야 합니다.")
    return result


def sequential_round_schedule(config: dict, registered_ids) -> dict:
    validate_sequential_round(config)
    clients = sorted(registered_ids)
    round_id = positive_int(config["round"], "round")
    seed = positive_int(config.get("seed", 42), "seed", 0)
    delayed = config.get("delay_clients", [])
    if (
        not isinstance(delayed, list)
        or any(type(client_id) is not int or client_id not in clients for client_id in delayed)
        or len(delayed) != len(set(delayed))
    ):
        raise ValueError("가상 지연 대상은 중복 없는 등록 클라이언트 ID여야 합니다.")
    timing = virtual_timing(config.get("virtual_timing"))
    count = max(1, round(len(clients) * config["participation_rate"]))
    selected = sorted(random.Random(seed + round_id).sample(clients, count))
    arrivals = {
        str(client_id): timing["delayed_arrival_seconds" if client_id in delayed else "normal_arrival_seconds"]
        for client_id in selected
    }
    deadline = timing["deadline_seconds"]
    last_arrival = max(arrivals.values())
    return {
        "round": round_id, "selected_clients": selected,
        "unselected_clients": sorted(set(clients) - set(selected)),
        "delay_clients": sorted(delayed), "virtual_timing": timing,
        "virtual_arrival_seconds": arrivals,
        "virtual_late_clients": [client_id for client_id in selected if arrivals[str(client_id)] > deadline],
        "virtual_deadline_seconds": deadline, "virtual_elapsed_seconds": min(last_arrival, deadline),
        "virtual_last_arrival_seconds": last_arrival, "deadline_rule": "arrival <= deadline",
    }
