import numpy as np
import pytest

from fluxoforum.scheduling import SafeExpression, ScheduleError, render_schedule


def test_linear_schedule():
    values = render_schedule("0:(1.0), 4:(2.0)", 5, 24)
    assert np.allclose(values, [1.0, 1.25, 1.5, 1.75, 2.0])


def test_expression_context():
    values = render_schedule("0:(1 + 0.1*sin(f*pi/2))", 4, 24)
    assert values[1] == pytest.approx(1.1)


@pytest.mark.parametrize(
    "expression",
    ["__import__('os')", "(1).__class__", "[x for x in [1]]", "open('x')"],
)
def test_expression_rejects_unsafe_syntax(expression):
    with pytest.raises(ScheduleError):
        SafeExpression(expression)

