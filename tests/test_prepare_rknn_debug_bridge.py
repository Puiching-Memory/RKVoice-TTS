from __future__ import annotations

from scripts.board.prepare_rknn_debug_bridge import (
    build_replace_adbd_command,
    build_start_rknn_server_command,
    socket_state_has_port,
)
from tests.test_support import WorkspaceTestCase


class PrepareRKNNDebugBridgeTests(WorkspaceTestCase):
    def test_socket_state_has_port_detects_listener(self) -> None:
        self.assertTrue(socket_state_has_port("LISTEN 0 0 127.0.0.1:5037 0.0.0.0:*", 5037))
        self.assertFalse(socket_state_has_port("LISTEN 0 0 0.0.0.0:5555 0.0.0.0:*", 5037))

    def test_build_replace_adbd_command_includes_backup_and_atomic_move(self) -> None:
        command = build_replace_adbd_command(
            staged_binary="/userdata/rkvoice-adbd/adbd",
            target_binary="/usr/bin/adbd",
            backup_path="/usr/bin/adbd.rkvoice.bak.20260413010101",
        )

        self.assertIn("cp '/usr/bin/adbd' '/usr/bin/adbd.rkvoice.bak.20260413010101'", command)
        self.assertIn("cp '/userdata/rkvoice-adbd/adbd' '/usr/bin/adbd.rkvoice.new'", command)
        self.assertIn("chmod +x '/usr/bin/adbd.rkvoice.new'", command)
        self.assertIn("mv -f '/usr/bin/adbd.rkvoice.new' '/usr/bin/adbd'", command)

    def test_build_start_rknn_server_command_exports_loglevel_and_log_path(self) -> None:
        command = build_start_rknn_server_command(
            remote_rknn_server_path="/usr/bin/rknn_server",
            loglevel=5,
            log_path="/userdata/rknn/server.log",
        )

        self.assertIn("mkdir -p '/userdata/rknn'", command)
        self.assertIn("pkill -x rknn_server", command)
        self.assertIn("export RKNN_SERVER_LOGLEVEL=5", command)
        self.assertIn("nohup '/usr/bin/rknn_server' >'/userdata/rknn/server.log' 2>&1 </dev/null &", command)