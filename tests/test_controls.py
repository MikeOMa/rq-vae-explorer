from rq_vae_explorer.ui.controls import (
    create_lambda_sliders,
    create_training_controls,
    format_health_text,
)


def test_format_health_text_empty():
    result = format_health_text([])
    assert "No data" in result


def test_format_health_text_with_data():
    health = [
        {"level": 1, "active": 14, "dead": 2, "total": 16},
        {"level": 2, "active": 12, "dead": 4, "total": 16},
    ]
    result = format_health_text(health)

    assert "Level 1" in result
    assert "14/16" in result
    assert "Level 2" in result
    assert "12/16" in result
