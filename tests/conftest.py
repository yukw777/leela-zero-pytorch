import pytest

from hydra._internal.hydra import GlobalHydra


@pytest.fixture
def clear_hydra():
    """
    This fixture clears GlobalHydra after the test is done.
    Use this when you're testing Hydra.
    """
    g = GlobalHydra()
    yield g
    g.clear()
