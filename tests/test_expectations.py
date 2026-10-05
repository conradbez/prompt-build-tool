"""YAML assertions must reject missing evidence and avoid coercing booleans."""
import pytest

from pbt.expectations import mismatches


@pytest.mark.parametrize('actual, expected, fragment', [
    ({}, {'ok': False}, 'missing field'),
    ({'ok': 0}, {'ok': False}, 'expected False'),
    ({'ok': 'false'}, {'ok': False}, 'expected False'),
    ([], [{'ref': 'D1', 'ok': False}], 'got 0'),
    ([{'ref': 'D1'}, {'ref': 'D1'}], [{'ref': 'D1'}], 'got 2'),
    ([1], [], 'expected 0 items'),
    (None, {}, 'expected a mapping'),
    ({}, [], 'expected a list'),
    ([1, 2], [2, 1], '[0]'),
])
def test_mismatch(actual, expected, fragment):
    assert fragment in '\n'.join(mismatches(actual, expected, 'model'))


def test_partial_nested_fields_and_ref_order():
    actual = {'parts': ['skip', {'ref': 'Q2', 'ok': True}, {'ref': 'D1', 'ok': False, 'notes': 'reversed'}], 'extra': 1}
    expected = {'parts': [{'ref': 'D1', 'ok': False}, {'ref': 'Q2', 'ok': True}]}
    assert mismatches(actual, expected, 'polarity') == []


def test_nested_null_and_empty_list_are_asserted():
    assert mismatches({'issue': None, 'issues': []}, {'issue': None, 'issues': []}, 'm') == []
    assert mismatches({'issue': 'fault'}, {'issue': None}, 'm')
