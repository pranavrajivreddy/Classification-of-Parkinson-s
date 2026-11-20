"""Private-data checks are opt-in; ordinary tests require no study data or API calls."""
import pytest

def pytest_addoption(parser):
    parser.addoption('--private-data', action='store_true', help='Run tests requiring locally authorized mPower artifacts')

def pytest_collection_modifyitems(config, items):
    if config.getoption('--private-data'):
        return
    for item in items:
        if 'private_data' in item.keywords:
            item.add_marker(pytest.mark.skip(reason='Requires authorized local study artifacts; use --private-data'))
