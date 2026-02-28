"""Tests for locomotion/teleop.py (8 tests)."""

from unittest.mock import MagicMock, patch, call
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from locomotion import teleop as teleop_mod  # conftest pre-imports mujoco so this is safe


# ---------------------------------------------------------------------------
# _TerminalInput tests
# ---------------------------------------------------------------------------

class TestTerminalInputInit:
    def test_terminal_input_init(self):
        ti = teleop_mod._TerminalInput()
        assert ti.enabled is False
        assert ti._fd is None
        assert ti._old_term is None


class TestTerminalInputEnter:
    def test_enter(self):
        ti = teleop_mod._TerminalInput()
        fake_fd = 0
        fake_old = [0] * 7  # termios attr list

        with patch.object(sys, "stdin") as mock_stdin, \
             patch.object(teleop_mod.termios, "tcgetattr", return_value=fake_old), \
             patch.object(teleop_mod.tty, "setcbreak") as mock_cbreak:

            mock_stdin.isatty.return_value = True
            mock_stdin.fileno.return_value = fake_fd

            result = ti.__enter__()

        assert result is ti
        assert ti.enabled is True
        mock_cbreak.assert_called_once_with(fake_fd)


class TestTerminalInputExit:
    def test_exit(self):
        ti = teleop_mod._TerminalInput()
        ti.enabled = True
        ti._fd = 0
        ti._old_term = [0] * 7

        with patch.object(teleop_mod.termios, "tcsetattr") as mock_tc:
            ti.__exit__(None, None, None)

        mock_tc.assert_called_once()
        assert ti.enabled is False


class TestReadArrowTail:
    def test_read_arrow_tail(self):
        ti = teleop_mod._TerminalInput()
        ti.enabled = True

        # Simulate reading "[D" (left arrow tail)
        read_calls = iter(["[", "D"])
        with patch.object(sys, "stdin") as mock_stdin, \
             patch.object(teleop_mod.select, "select") as mock_select:

            mock_stdin.read = lambda n: next(read_calls)
            # Both select calls return readable
            mock_select.return_value = ([mock_stdin], [], [])

            result = ti._read_arrow_tail()

        assert result == "[D"


class TestReadKeys:
    def test_read_keys(self):
        # Disabled returns empty
        ti = teleop_mod._TerminalInput()
        assert ti.read_keys() == []

        # Enabled returns chars
        ti.enabled = True
        chars = iter(["w", ""])
        select_results = iter([
            ([True], [], []),  # first call: readable
            ([], [], []),       # second call: nothing
        ])
        with patch.object(teleop_mod.select, "select", side_effect=select_results), \
             patch.object(sys, "stdin") as mock_stdin:
            mock_stdin.read = lambda n: next(chars)
            keys = ti.read_keys()

        assert keys == ["w"]


# ---------------------------------------------------------------------------
# build_obs test
# ---------------------------------------------------------------------------

class TestBuildObs:
    def test_build_obs(self):
        from locomotion.constants import OBS_SIZE, TOTAL_OBS, NUM_ACTUATORS

        data = SimpleNamespace(
            xmat=np.tile(np.eye(3).flatten(), (5, 1)),  # identity rotation per body
            cvel=np.zeros((5, 6)),                       # (rot, lin) per body
            qpos=np.zeros(16, dtype=np.float64),
            qvel=np.zeros(15, dtype=np.float64),
        )
        # Set joint positions to default pose
        data.qpos[7:16] = teleop_mod.DEFAULT_POSE

        base_body_id = 1
        command = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        last_action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
        obs_history = np.zeros(TOTAL_OBS, dtype=np.float32)

        obs = teleop_mod.build_obs(data, base_body_id, command, last_action, obs_history)
        assert obs.shape == (TOTAL_OBS,)

        # First OBS_SIZE elements should be set (not all zero due to gravity)
        assert not np.allclose(obs[:OBS_SIZE], 0.0)

        # Second call should stack (history rolls)
        obs2 = teleop_mod.build_obs(data, base_body_id, command, last_action, obs)
        assert not np.allclose(obs2[OBS_SIZE:2 * OBS_SIZE], 0.0)

        # All values within clip bounds
        assert np.all(obs2 >= -100.0) and np.all(obs2 <= 100.0)


# ---------------------------------------------------------------------------
# main() test
# ---------------------------------------------------------------------------

class TestMain:
    def test_main(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", [
            "teleop.py",
            "--no-policy",
            "--xml-path", "dummy.xml",
        ])

        mock_model = MagicMock()
        mock_model.opt.timestep = 0.004
        mock_model.dof_damping = np.zeros(15)
        mock_data = MagicMock()
        mock_data.qpos = np.zeros(16, dtype=np.float64)
        mock_data.qvel = np.zeros(15, dtype=np.float64)
        mock_data.xmat = np.tile(np.eye(3).flatten(), (5, 1))
        mock_data.cvel = np.zeros((5, 6))
        mock_data.ctrl = np.zeros(9)

        mock_viewer_ctx = MagicMock()
        mock_viewer_ctx.is_running.return_value = False

        mock_from_xml = MagicMock(return_value=mock_model)
        mock_term_inst = MagicMock()
        mock_term_inst.read_keys.return_value = []

        with patch.object(teleop_mod.mujoco.MjModel, "from_xml_path", mock_from_xml), \
             patch.object(teleop_mod.mujoco, "MjData", return_value=mock_data), \
             patch.object(teleop_mod.mujoco, "mj_name2id", side_effect=[1, 0]), \
             patch.object(teleop_mod.mujoco, "mj_resetDataKeyframe"), \
             patch.object(teleop_mod.mujoco, "mj_forward"), \
             patch.object(teleop_mod.mujoco, "viewer") as mock_viewer_mod, \
             patch.object(teleop_mod, "_TerminalInput") as mock_term_cls:

            mock_viewer_mod.launch_passive.return_value.__enter__ = MagicMock(return_value=mock_viewer_ctx)
            mock_viewer_mod.launch_passive.return_value.__exit__ = MagicMock(return_value=False)
            mock_term_cls.return_value.__enter__ = MagicMock(return_value=mock_term_inst)
            mock_term_cls.return_value.__exit__ = MagicMock(return_value=False)

            teleop_mod.main()

        mock_from_xml.assert_called_once_with("dummy.xml")


class TestPause:
    def test_pause_skips_physics(self, monkeypatch, capsys):
        """Pressing 'p' toggles pause; physics/policy are skipped while paused."""
        monkeypatch.setattr(sys, "argv", [
            "teleop.py", "--no-policy", "--xml-path", "dummy.xml",
        ])

        mock_model = MagicMock()
        mock_model.opt.timestep = 0.004
        mock_model.dof_damping = np.zeros(15)
        mock_data = MagicMock()
        mock_data.qpos = np.zeros(16, dtype=np.float64)
        mock_data.qvel = np.zeros(15, dtype=np.float64)
        mock_data.xmat = np.tile(np.eye(3).flatten(), (5, 1))
        mock_data.cvel = np.zeros((5, 6))
        mock_data.ctrl = np.zeros(9)

        # Run 5 iterations: all paused (default), then viewer stops
        run_count = [0]
        def is_running():
            run_count[0] += 1
            return run_count[0] <= 5
        mock_viewer_ctx = MagicMock()
        mock_viewer_ctx.is_running.side_effect = is_running

        mock_from_xml = MagicMock(return_value=mock_model)
        mock_term_inst = MagicMock()
        # No keys pressed — stays paused the whole time
        mock_term_inst.read_keys.return_value = []

        fake_time = [0.0]
        def advancing_clock():
            fake_time[0] += 1.0
            return fake_time[0]
        monkeypatch.setattr(teleop_mod.time, "perf_counter", advancing_clock)
        monkeypatch.setattr(teleop_mod.time, "sleep", lambda _: None)

        with patch.object(teleop_mod.mujoco.MjModel, "from_xml_path", mock_from_xml), \
             patch.object(teleop_mod.mujoco, "MjData", return_value=mock_data), \
             patch.object(teleop_mod.mujoco, "mj_name2id", side_effect=[1, 0]), \
             patch.object(teleop_mod.mujoco, "mj_resetDataKeyframe"), \
             patch.object(teleop_mod.mujoco, "mj_forward"), \
             patch.object(teleop_mod.mujoco, "mj_step") as mock_step, \
             patch.object(teleop_mod.mujoco, "viewer") as mock_viewer_mod, \
             patch.object(teleop_mod, "_TerminalInput") as mock_term_cls:

            mock_viewer_mod.launch_passive.return_value.__enter__ = MagicMock(return_value=mock_viewer_ctx)
            mock_viewer_mod.launch_passive.return_value.__exit__ = MagicMock(return_value=False)
            mock_term_cls.return_value.__enter__ = MagicMock(return_value=mock_term_inst)
            mock_term_cls.return_value.__exit__ = MagicMock(return_value=False)

            teleop_mod.main()

        # Physics should never have been stepped (simulation starts paused)
        mock_step.assert_not_called()
        # Viewer sync should still have been called (camera updates while paused)
        mock_viewer_ctx.sync.assert_called()
        # Output should show initial [paused] message
        output = capsys.readouterr().out
        assert "[paused]" in output


class TestDiagnostics:
    def test_diagnostics_printed_at_one_second(self, monkeypatch, capsys):
        """Diagnostics line is printed after 50 steps (1 second at 50 Hz)."""
        monkeypatch.setattr(sys, "argv", [
            "teleop.py", "--no-policy", "--xml-path", "dummy.xml",
        ])

        mock_model = MagicMock()
        mock_model.opt.timestep = 0.004
        mock_model.dof_damping = np.zeros(15)
        mock_data = MagicMock()
        mock_data.qpos = np.zeros(16, dtype=np.float64)
        mock_data.qvel = np.zeros(15, dtype=np.float64)
        mock_data.xmat = np.tile(np.eye(3).flatten(), (5, 1))
        mock_data.cvel = np.zeros((5, 6))
        mock_data.ctrl = np.zeros(9)

        # Viewer runs for 52 iterations: 1 to unpause + 51 for 50 physics steps
        run_count = [0]
        def is_running():
            run_count[0] += 1
            return run_count[0] <= 52
        mock_viewer_ctx = MagicMock()
        mock_viewer_ctx.is_running.side_effect = is_running

        mock_from_xml = MagicMock(return_value=mock_model)
        mock_term_inst = MagicMock()
        # First call returns "p" to unpause, then empty
        key_calls = iter([["p"]])
        mock_term_inst.read_keys.side_effect = lambda: next(key_calls, [])

        # Incrementing time so we never hit the sleep/continue branch
        fake_time = [0.0]
        def advancing_clock():
            fake_time[0] += 1.0
            return fake_time[0]
        monkeypatch.setattr(teleop_mod.time, "perf_counter", advancing_clock)
        monkeypatch.setattr(teleop_mod.time, "sleep", lambda _: None)

        with patch.object(teleop_mod.mujoco.MjModel, "from_xml_path", mock_from_xml), \
             patch.object(teleop_mod.mujoco, "MjData", return_value=mock_data), \
             patch.object(teleop_mod.mujoco, "mj_name2id", side_effect=[1, 0]), \
             patch.object(teleop_mod.mujoco, "mj_resetDataKeyframe"), \
             patch.object(teleop_mod.mujoco, "mj_forward"), \
             patch.object(teleop_mod.mujoco, "mj_step"), \
             patch.object(teleop_mod.mujoco, "viewer") as mock_viewer_mod, \
             patch.object(teleop_mod, "_TerminalInput") as mock_term_cls:

            mock_viewer_mod.launch_passive.return_value.__enter__ = MagicMock(return_value=mock_viewer_ctx)
            mock_viewer_mod.launch_passive.return_value.__exit__ = MagicMock(return_value=False)
            mock_term_cls.return_value.__enter__ = MagicMock(return_value=mock_term_inst)
            mock_term_cls.return_value.__exit__ = MagicMock(return_value=False)

            teleop_mod.main()

        output = capsys.readouterr().out
        assert "[diag t=1s step=50]" in output
        assert "obs:" in output
        assert "action:" in output
