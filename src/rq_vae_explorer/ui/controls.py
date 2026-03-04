"""Formatting helpers shared between the API and tests."""


def format_health_text(health: list[dict[str, int]]) -> str:
    """Format codebook health statistics as plain text.

    Args:
        health: List of dicts with level, active, dead, total keys

    Returns:
        Formatted text string
    """
    if not health:
        return "No data yet"

    lines = ["Codebook Health"]
    for h in health:
        status = "OK" if h["dead"] == 0 else "WARN"
        lines.append(f"{status} Level {h['level']}: {h['active']}/{h['total']} active")
        if h["dead"] > 0:
            lines.append(f"   ({h['dead']} dead)")

    return "\n".join(lines)


def format_step_text(step: int, is_training: bool) -> str:
    """Format current step and training status.

    Args:
        step: Current training step
        is_training: Whether training is active

    Returns:
        Formatted text string
    """
    status = "Training" if is_training else "Paused"
    return f"Step: {step:,} | {status}"
