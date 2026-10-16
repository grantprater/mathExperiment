import pytest

from mathexp import guard


def pytest_configure(config):
    """Tests run at idle priority and single-threaded, like the CLI."""
    guard.apply()


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Every test gets its own truth cache; the repo cache/ is never touched."""
    monkeypatch.setenv("MATHEXP_CACHE", str(tmp_path / "truth.json"))
