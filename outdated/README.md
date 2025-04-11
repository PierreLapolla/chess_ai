# Chess AI

This project is a framework to develop chess AI using Python.

## Installation

### Requirements

- [UV](https://docs.astral.sh/uv/)
- Project name in [pyproject.toml](pyproject.toml) file must be the same as your module's name in `src` folder.


### Sync Dependencies

```bash
  uv sync
```

## Running the project

To run the project, run the following command:

```bash
  uv run python -m src.chess_ai
```

## Running Tests

To run tests, run the following command:

```bash
  uv run pytest
```

## Linting and formatting

We can use UV tools to lint and format the code.

```bash
  uvx ruff check . --fix
  uvx ruff format .
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
