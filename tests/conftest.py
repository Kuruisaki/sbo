def pytest_configure(config):
    config.addinivalue_line("markers", "integration: launches the real configured XFOIL executable")
