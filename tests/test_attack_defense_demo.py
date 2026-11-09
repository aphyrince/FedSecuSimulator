import torch

from experiments.attack_defense_demo import (
    ATTACKER_IDS,
    CLIP_MAX_NORM,
    aggregate,
    distance_between,
    make_updates,
    run_experiment,
    update_l2_norm,
)
from security.defenses import clip_update


def test_malicious_clients_submit_same_minus_five_update():
    clean, attacked, _ = make_updates()
    expected = -5 * torch.stack([clean[i][0] for i in ATTACKER_IDS]).mean(0)
    for i in ATTACKER_IDS:
        assert torch.allclose(attacked[i][0], expected)
    for i in range(10):
        if i not in ATTACKER_IDS:
            assert torch.equal(attacked[i][0], clean[i][0])


def test_clipping_caps_attacker_updates():
    _, attacked, _ = make_updates()
    for update in attacked:
        assert update_l2_norm(clip_update(update, CLIP_MAX_NORM)) <= CLIP_MAX_NORM + 1e-9


def test_defense_improves_distance_in_this_toy_example():
    clean, attacked, counts = make_updates()
    reference = aggregate(clean, counts, 'fedavg')
    attacked_fedavg = aggregate(attacked, counts, 'fedavg')
    clipped_fedavg = aggregate(attacked, counts, 'clipping_fedavg')
    clipped_median = aggregate(attacked, counts, 'clipping_median')
    assert distance_between(clipped_fedavg, reference) < distance_between(attacked_fedavg, reference)
    assert distance_between(clipped_median, reference) < distance_between(attacked_fedavg, reference)


def test_six_scenarios_and_twenty_client_logs():
    summary, clients = run_experiment()
    assert len(summary) == 6
    assert len(clients) == 20
    assert {(r['mode'], r['defense']) for r in summary} == {
        (mode, defense)
        for mode in ('normal', 'attack')
        for defense in ('fedavg', 'clipping_fedavg', 'clipping_median')
    }
