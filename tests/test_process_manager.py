import subprocess
import signal
import threading
import unittest
from unittest.mock import Mock, patch

from exp_manager.app import _register_sigterm_handler
from exp_manager.services.process_manager import ProcessManager


class ProcessManagerTests(unittest.TestCase):
    def test_unexpected_exit_records_signal_but_cleanup_does_not(self):
        process = Mock(pid=123)
        process.wait.return_value = -signal.SIGSEGV
        manager = ProcessManager("127.0.0.1", 1, 5000, 5101)
        manager.client_processes[0] = process

        with self.assertLogs("fedsec.runtime", level="ERROR") as logs:
            manager._watch_process(process, 0)
        self.assertIn("client_id=0 child_pid=123", logs.output[0])
        self.assertIn("signal=SIGSEGV", logs.output[0])

        manager.client_processes.clear()
        with self.assertNoLogs("fedsec.runtime", level="ERROR"):
            manager._watch_process(process, 0)

    def test_stop_waits_after_terminate_and_kill(self):
        process = Mock()
        process.wait.side_effect = [
            subprocess.TimeoutExpired("client", 2),
            subprocess.TimeoutExpired("client", 2),
            None,
        ]
        process.poll.return_value = None
        manager = ProcessManager("127.0.0.1", 1, 5000, 5101)
        manager.client_processes[0] = process

        with patch.object(manager, "_post"):
            manager.stop()

        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(process.wait.call_count, 3)
        self.assertNotIn(0, manager.client_processes)

    def test_sigterm_stops_children_before_server(self):
        children_stopped = threading.Event()
        server_stopped = threading.Event()
        process_manager = Mock()
        process_manager.stop.side_effect = children_stopped.set
        http_server = Mock()
        http_server.shutdown.side_effect = server_stopped.set

        with patch("exp_manager.app.signal.signal") as register_signal:
            _register_sigterm_handler(process_manager, http_server)
        handler = register_signal.call_args.args[1]
        handler(None, None)

        self.assertTrue(children_stopped.wait(timeout=2))
        self.assertTrue(server_stopped.wait(timeout=2))
        process_manager.stop.assert_called_once()
        http_server.shutdown.assert_called_once()
