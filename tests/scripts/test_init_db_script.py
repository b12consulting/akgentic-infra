"""Tests for the ``akgentic.infra.scripts.init_db`` init-container entry point.

The exit paths run without a database: the registry module's ``init_db`` is
replaced where a call would reach one.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

import pytest

import akgentic.infra.scripts.init_db as script
from akgentic.infra.adapters.shared import postgres_channel_registry

_ENV_VAR = "DB_CONN_STRING_PERSISTENCE"
DSN = "postgresql://akgentic:s3cret-pw@localhost:5432/akgentic"


class _InitDbSpy:
    def __init__(self, error: BaseException | None = None) -> None:
        self.calls: list[str] = []
        self._error = error

    def __call__(self, dsn: str) -> None:
        self.calls.append(dsn)
        if self._error is not None:
            raise self._error


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> _InitDbSpy:
    spy = _InitDbSpy()
    monkeypatch.setattr(postgres_channel_registry, "init_db", spy)
    return spy


def test_unset_variable_returns_2(monkeypatch: pytest.MonkeyPatch, spy: _InitDbSpy) -> None:
    monkeypatch.delenv(_ENV_VAR, raising=False)
    assert script.main() == 2
    assert spy.calls == []


def test_non_postgresql_scheme_returns_2_without_logging_the_dsn(
    monkeypatch: pytest.MonkeyPatch, spy: _InitDbSpy, caplog: pytest.LogCaptureFixture
) -> None:
    """Nagra's error for any other scheme echoes the DSN, so the script refuses it first."""
    monkeypatch.setenv(_ENV_VAR, "postgres://u:pw-secret@h/db")
    with caplog.at_level(logging.DEBUG):
        assert script.main() == 2
    assert spy.calls == []
    assert caplog.records
    assert all("pw-secret" not in record.getMessage() for record in caplog.records)
    assert "pw-secret" not in caplog.text


def test_valid_dsn_calls_init_db_once_and_returns_0(
    monkeypatch: pytest.MonkeyPatch, spy: _InitDbSpy
) -> None:
    monkeypatch.setenv(_ENV_VAR, DSN)
    assert script.main() == 0
    assert spy.calls == [DSN]


@pytest.mark.parametrize(
    "error", [ImportError("No module named 'nagra'"), RuntimeError("connection refused")]
)
def test_a_failing_init_db_returns_1(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    spy = _InitDbSpy(error)
    monkeypatch.setattr(postgres_channel_registry, "init_db", spy)
    monkeypatch.setenv(_ENV_VAR, DSN)
    assert script.main() == 1
    assert spy.calls == [DSN]


def test_the_import_error_names_the_extra(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        postgres_channel_registry, "init_db", _InitDbSpy(ImportError("No module named 'nagra'"))
    )
    monkeypatch.setenv(_ENV_VAR, DSN)
    with caplog.at_level(logging.ERROR):
        assert script.main() == 1
    assert 'pip install "akgentic-infra[postgres]"' in caplog.text


def test_python_dash_m_without_the_variable_exits_2() -> None:
    env = {k: v for k, v in os.environ.items() if k != _ENV_VAR}
    result = subprocess.run(
        [sys.executable, "-m", "akgentic.infra.scripts.init_db"],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert result.returncode == 2, f"stdout={result.stdout!r} stderr={result.stderr!r}"
