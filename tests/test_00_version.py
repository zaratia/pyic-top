import pyic_top


def test_version() -> None:
    assert pyic_top.__version__ != "999"
