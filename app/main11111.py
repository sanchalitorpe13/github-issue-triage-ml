cbrain_cli/
■■■ __init__.py
■■■ cli.py # Root Click group, global --output / --verbose flags
■■■ auth.py # login, logout, whoami commands
■■■ files.py # files sub-command group
■■■ tasks.py # tasks sub-command group
■■■ projects.py # projects, groups commands
■■■ tools.py # tools, data-providers commands
■■■ client/
■ ■■■ api.py # CbrainClient — all HTTP interactions
■ ■■■ models.py # Dataclasses for API response objects
■■■ config.py # Config read/write, keyring integration
■■■ output.py # Rich formatters, JSON output
■■■ exceptions.py # CLI exception hierarchy
tests/
■■■ test_auth.py
■■■ test_files.py
■■■ test_tasks.py
■■■ conftest.py # Fixtures, mock CbrainClient
docs/
■■■ quickstart.rst
■■■ commands/
■■■ conf.py