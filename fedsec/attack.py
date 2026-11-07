"""Arithmetic-mean collusion and a transport-independent peer inbox."""

from dataclasses import dataclass
import torch
from .config import AttackConfig, integer
from .contracts import ClientUpdate
from .tensors import validate_state


@dataclass
class AttackOutcome:
    updates: list[ClientUpdate]
    applied: bool
    reason: str


class PeerUpdateBuffer:
    """One inbox per experiment/round/model. Role 3 handles waiting/timeouts.

    expected_clients includes this client itself. Add its clean update first,
    exchange only clean updates, then finalize when all arrive or at the deadline.
    """

    def __init__(self, own_update: ClientUpdate, expected_clients: list[int] | tuple[int, ...]):
        if not expected_clients or len(set(expected_clients)) != len(expected_clients):
            raise ValueError("협력 참여 ID는 중복 없는 목록이어야 합니다.")
        for client_id in expected_clients:
            integer("expected_clients", client_id)
        if own_update.client_id not in expected_clients:
            raise ValueError("협력 목록에 자기 ID가 없습니다.")
        self.expected = tuple(sorted(expected_clients))
        self.reference = own_update
        self.received: dict[int, ClientUpdate] = {}
        self.add(own_update)

    def add(self, update: ClientUpdate) -> None:
        reference = self.reference
        if update.client_id not in self.expected:
            raise ValueError("이번 라운드의 협력 대상이 아닙니다.")
        if update.client_id in self.received:
            raise ValueError("협력 업데이트를 중복 수신했습니다.")
        if (update.experiment_id, update.round_id, update.model_version, update.base_model_hash) != (
                reference.experiment_id, reference.round_id, reference.model_version, reference.base_model_hash):
            raise ValueError("협력 업데이트의 실험/라운드/시작 모델이 다릅니다.")
        validate_state(update.delta, reference.delta)
        self.received[update.client_id] = update.with_delta(update.delta)

    def status(self) -> dict:
        return {"experiment_id": self.reference.experiment_id, "round_id": self.reference.round_id,
                "received_clients": sorted(self.received), "expected_count": len(self.expected),
                "completed": len(self.received) == len(self.expected)}

    def finalize(self, config: AttackConfig) -> AttackOutcome:
        updates = [self.received[i] for i in sorted(self.received)]
        if not config.enabled:
            return AttackOutcome(updates, False, "disabled")
        if not set(self.expected).issubset(config.malicious_clients):
            raise ValueError("협력 대상에 정상 클라이언트가 포함되었습니다.")
        missing = set(self.expected) - set(self.received)
        if missing and config.missing_peer_policy == "error":
            raise TimeoutError(f"협력 업데이트 누락: {sorted(missing)}")
        if missing and config.missing_peer_policy == "skip_attack":
            return AttackOutcome(updates, False, "missing_peers")
        if len(updates) < config.min_colluders:
            return AttackOutcome(updates, False, "insufficient_colluders")
        delta = {}
        for name, tensor in self.reference.delta.items():
            average = torch.zeros_like(tensor, dtype=torch.float64, device="cpu")
            for update in updates:
                average.add_(update.delta[name].detach().cpu().double(), alpha=1 / len(updates))
            delta[name] = (average * config.scale).to(tensor.dtype)
        validate_state(delta, self.reference.delta)
        return AttackOutcome([update.with_delta(delta) for update in updates], True, "collusion")


def apply_collusion(clean_updates: list[ClientUpdate], config: AttackConfig,
                    participating_clients: tuple[int, ...]) -> AttackOutcome:
    """Standalone orchestration; real clients each own a PeerUpdateBuffer."""
    ids = [update.client_id for update in clean_updates]
    if len(set(ids)) != len(ids) or not set(ids).issubset(participating_clients):
        raise ValueError("정상 업데이트에 중복 또는 비참여 ID가 있습니다.")
    expected = sorted(set(config.malicious_clients) & set(participating_clients))
    malicious = [update for update in clean_updates if update.client_id in expected]
    if not config.enabled or not malicious:
        return AttackOutcome(list(clean_updates), False, "disabled_or_no_malicious_update")
    inbox = PeerUpdateBuffer(malicious[0], expected)
    for update in malicious[1:]:
        inbox.add(update)
    outcome = inbox.finalize(config)
    replacements = {update.client_id: update for update in outcome.updates}
    return AttackOutcome([replacements.get(update.client_id, update) for update in clean_updates],
                         outcome.applied, outcome.reason)
