import pytest
from Module_05.ex_01.TinyStatistician import TinyStatistician

MEAN_ARGS = [
    (),
]


@pytest.mark.parametrize("args, expected", MEAN_ARGS)
def test_mean(args, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            mean = TinyStatistician().mean(args)
    else:
        mean = TinyStatistician().mean(args)
        assert mean == expected


MEDIAN_ARGS = [
    (),
]

QUARTILE_ARGS = [
    (),
]

PERCENTILE_ARGS = [
    (),
]

VAR_ARGS = [
    (),
]

STD_ARGS = [
    (),
]
