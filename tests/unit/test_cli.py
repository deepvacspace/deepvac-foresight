"""Unit tests for deepvac/cli.py's dispatcher: argv forwarding, --list/--help,
and unknown-command handling."""

from __future__ import annotations

import sys

import pytest
from deepvac import cli


def test_every_command_module_is_importable():
    """Every entry in COMMANDS must resolve to a real, importable module --
    catches a typo'd module path before it ships as a broken subcommand."""
    import importlib

    for entry in cli.COMMANDS.values():
        module_name = entry[0]
        importlib.import_module(module_name)


def test_list_flag_prints_every_command(capsys):
    exit_code = cli.main(["--list"])
    out = capsys.readouterr().out
    assert exit_code == 0
    for command in cli.COMMANDS:
        assert command in out


def test_no_args_behaves_like_list(capsys):
    exit_code = cli.main([])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Available deepvac commands" in out


def test_unknown_command_prints_list_and_returns_error_code(capsys):
    exit_code = cli.main(["not-a-real-command"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Unknown command" in captured.err
    assert "Available deepvac commands" in captured.out


def test_dispatch_forwards_argv_to_target_module(monkeypatch):
    """--help on a dispatched command should behave exactly like invoking
    that module directly (deepvac/cli.py's whole design premise)."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["train-gru", "--help"])
    # argparse's own --help handling exits 0.
    assert exc_info.value.code == 0


def test_dispatch_sets_sys_argv_to_look_like_direct_invocation(monkeypatch):
    """Registers a fake module directly in sys.modules (found by the real
    importlib.import_module via its own cache) rather than patching
    importlib itself, which would risk affecting unrelated imports."""
    captured = {}

    class FakeModule:
        @staticmethod
        def main():
            captured["argv"] = list(sys.argv)
            return 0

    monkeypatch.setitem(cli.COMMANDS, "fake-command", ("fake_module_for_test", "test only"))
    monkeypatch.setitem(sys.modules, "fake_module_for_test", FakeModule())

    cli.main(["fake-command", "--foo", "bar"])
    assert captured["argv"] == ["fake_module_for_test", "--foo", "bar"]
