"""No training/forward/backward; fixed tensors verify the protocol arithmetic."""

from dataclasses import replace
import unittest
import torch
from fedsec.aggregation import fedavg
from fedsec.artifacts import decode_update, encode_update
from fedsec.attack import PeerUpdateBuffer, apply_collusion
from fedsec.config import AttackConfig
from fedsec.contracts import ClientUpdate, RoundContext, TrainRequest
from fedsec.tensors import clone_state, state_hash


class AlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.state = {"weight": torch.tensor([10.0, 20.0])}
        self.context = RoundContext("exp", 1, 0, (0, 2, 7))

    def update(self, client_id, values=(1.0, 2.0), samples=2):
        return ClientUpdate("exp", client_id, 1, 0, samples,
                            state_hash(self.state), {"weight": torch.tensor(values)})

    def test_fedavg_sample_weighting_and_input_preservation(self):
        original = clone_state(self.state)
        updates = [self.update(0, (1.0, 2.0), 1), self.update(2, (3.0, 6.0), 3)]
        result = fedavg(self.state, updates, self.context, sample_counts={0: 1, 2: 3})
        torch.testing.assert_close(result["weight"], torch.tensor([12.5, 25.0]))
        torch.testing.assert_close(self.state["weight"], original["weight"], rtol=0, atol=0)
        result["weight"].zero_()
        torch.testing.assert_close(self.state["weight"], original["weight"], rtol=0, atol=0)

    def test_only_received_samples_are_in_denominator(self):
        result = fedavg(self.state, [self.update(2)], self.context)
        torch.testing.assert_close(result["weight"], torch.tensor([11.0, 22.0]))

    def test_empty_duplicate_unselected_or_wrong_samples_rejected(self):
        update = self.update(0)
        invalid = ([], [update, update], [self.update(4)])
        for updates in invalid:
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                fedavg(self.state, updates, self.context)
        with self.assertRaises(ValueError):
            fedavg(self.state, [update], self.context, sample_counts={0: 999})

    def test_stale_context_or_different_base_model_rejected(self):
        update = self.update(0)
        for modified in (replace(update, experiment_id="other"), replace(update, round_id=2),
                         replace(update, model_version=1), replace(update, base_model_hash="0" * 64)):
            with self.subTest(update=modified), self.assertRaises(ValueError):
                fedavg(self.state, [modified], self.context)

    def test_tensor_schema_and_nonfinite_values_rejected(self):
        for delta in ({"other": torch.ones(2)}, {"weight": torch.ones(3)},
                      {"weight": torch.ones(2, dtype=torch.float64)}):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                fedavg(self.state, [replace(self.update(0), delta=delta)], self.context)
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.update(0, (value, 0.0))

    def test_collusion_is_unweighted_mean_times_scale(self):
        normal = self.update(0)
        first = self.update(2, (2.0, 4.0), samples=1)
        second = self.update(7, (4.0, 8.0), samples=9)
        outcome = apply_collusion([normal, first, second], AttackConfig(), self.context.selected_clients)
        self.assertTrue(outcome.applied)
        by_id = {update.client_id: update for update in outcome.updates}
        torch.testing.assert_close(by_id[2].delta["weight"], torch.tensor([-15.0, -30.0]))
        torch.testing.assert_close(by_id[7].delta["weight"], torch.tensor([-15.0, -30.0]))
        torch.testing.assert_close(by_id[0].delta["weight"], normal.delta["weight"])
        torch.testing.assert_close(first.delta["weight"], torch.tensor([2.0, 4.0]))
        self.assertEqual(by_id[2].num_samples, 1)
        self.assertEqual(by_id[7].num_samples, 9)
        self.assertNotIn("malicious", by_id[2].metadata())

    def test_nonparticipating_attackers_are_not_expected(self):
        own = self.update(2)
        outcome = apply_collusion([own], AttackConfig(), (2,))
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "insufficient_colluders")
        torch.testing.assert_close(outcome.updates[0].delta["weight"], own.delta["weight"])

    def test_missing_peer_policies(self):
        own = self.update(2)
        inbox = PeerUpdateBuffer(own, [2, 7])
        self.assertEqual(inbox.status()["received_clients"], [2])
        self.assertFalse(inbox.status()["completed"])
        self.assertEqual(inbox.finalize(AttackConfig()).reason, "missing_peers")
        with self.assertRaises(TimeoutError):
            inbox.finalize(AttackConfig(missing_peer_policy="error"))
        outcome = inbox.finalize(AttackConfig(missing_peer_policy="available_only", min_colluders=1))
        torch.testing.assert_close(outcome.updates[0].delta["weight"], torch.tensor([-5.0, -10.0]))

    def test_disabling_attack_while_waiting_returns_clean_updates(self):
        own = self.update(2)
        inbox = PeerUpdateBuffer(own, [2, 7])
        outcome = inbox.finalize(AttackConfig(enabled=False, malicious_clients=()))
        self.assertFalse(outcome.applied)
        self.assertEqual(outcome.reason, "disabled")
        torch.testing.assert_close(outcome.updates[0].delta["weight"], own.delta["weight"])

    def test_peer_inbox_rejects_duplicate_and_wrong_round(self):
        inbox = PeerUpdateBuffer(self.update(2), [2, 7])
        with self.assertRaises(ValueError):
            inbox.add(self.update(2))
        with self.assertRaises(ValueError):
            inbox.add(replace(self.update(7), round_id=2))
        inbox.add(self.update(7))
        self.assertTrue(inbox.status()["completed"])

    def test_binary_round_trip_and_definition(self):
        update = self.update(2)
        decoded = decode_update(encode_update(update))
        self.assertEqual(decoded.metadata(), update.metadata())
        torch.testing.assert_close(decoded.delta["weight"], update.delta["weight"], rtol=0, atol=0)
        decoded.delta["weight"].zero_()
        self.assertGreater(update.delta["weight"].sum().item(), 0)

    def test_api_request_uses_supplied_hyperparameters_and_context(self):
        payload = {"round_id": 2, "model_version": 1, "local_epochs": 4,
                   "batch_size": 17, "learning_rate": 0.012, "model_url": "/models/global.pt"}
        request = TrainRequest.from_api(payload, experiment_id="exp")
        self.assertEqual((request.local_epochs, request.batch_size, request.learning_rate), (4, 17, 0.012))
        with self.assertRaises(ValueError):
            TrainRequest.from_api({**payload, "experiment_id": "other"}, experiment_id="exp")
        with self.assertRaises(ValueError):
            TrainRequest.from_api({**payload, "batch_size": True}, experiment_id="exp")


if __name__ == "__main__":
    unittest.main()
