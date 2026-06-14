from fluxoforum.model import select_loader


def test_loader_prefers_fp8_on_modern_gpu():
    decision = select_loader("auto", True, (8, 9))
    assert decision.selected == "fp8"


def test_loader_falls_back_on_old_gpu():
    decision = select_loader("auto", True, (7, 5))
    assert decision.selected == "bf16"


def test_loader_uses_bf16_on_ampere():
    decision = select_loader("auto", True, (8, 6))
    assert decision.selected == "bf16"
