# Project Title

Project description

## Installation

### Requirements

- [UV](https://docs.astral.sh/uv/)
- Project name in [pyproject.toml](pyproject.toml) file must be the same as your module's name in `src` folder.

### Initialize the project:

```bash
  uv sync
  uv run pre-commit install
```

## Running the project

To run the project locally, run the following command:

```bash
  uv run python -m src.lightning_template
```

## Tests, linting and formatting - AUTOMATED

These steps are automatically handled by [pre-commit](https://pre-commit.com/) hooks defined
in [.pre-commit-config.yaml](.pre-commit-config.yaml).
Additionally, we can run them manually with the following commands:

```bash
  uv run pytest
```

```bash
  uvx ruff check . --fix
```

```bash
  uvx ruff format .
```

You can also run pre-commit hooks manually with:

```bash
  pre-commit run --all-files
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
