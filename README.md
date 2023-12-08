# CLI-OAI

[![Build](https://github.com/cmi-dair/cli-oai/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/cmi-dair/cli-oai/actions/workflows/test.yaml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/cmi-dair/cli-oai/branch/main/graph/badge.svg?token=22HWWFWPW5)](https://codecov.io/gh/cmi-dair/cli-oai)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![stability-wip](https://img.shields.io/badge/stability-work_in_progress-lightgrey.svg)
[![MIT License](https://img.shields.io/badge/license-LGPL_2.1-blue.svg)](https://github.com/cmi-dair/cli-oai/blob/main/LICENSE)
[![pages](https://img.shields.io/badge/api-docs-blue)](https://cmi-dair.github.io/cli-oai)

CLI-OAI is a Python command-line interface for interacting with the OpenAI API.

## Usage

Before running oai, make sure the environment variable OPENAI_API_KEY is set to
your API key. To use the CLI, run `oai --help` in your terminal. See the `--help`
function of the subcommands for more information on each command.

## Installation

Get the newest development version via:

```sh
poetry add git+https://github.com/cmi-dair/cli-oai
```

## Contributing

Please see the [contributing guidelines](CONTRIBUTING.md) for guidelines on contributing to this project.

## License

This project is licensed under the terms of the [L-GPLv2.1 license](LICENSE).
