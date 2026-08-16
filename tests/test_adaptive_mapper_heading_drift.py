from mapper.AdaptiveHeadingSafety import classify_forward_heading_drift


def test_moderate_post_turn_drift_is_recoverable() -> None:
    assert (
        classify_forward_heading_drift(
            11.0,
            nominal_limit=8.0,
            recoverable_limit=20.0,
        )
        == "recoverable"
    )


def test_large_forward_heading_drift_remains_unsafe() -> None:
    assert (
        classify_forward_heading_drift(
            -24.0,
            nominal_limit=8.0,
            recoverable_limit=20.0,
        )
        == "unsafe"
    )
