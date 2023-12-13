# CLI-OAI

CLI-OAI is a Python command-line interface for interacting with the OpenAI API. It provides a set of commands to interact with various OpenAI services such as Speech-to-Text (STT), Text-to-Speech (TTS), and Image Generation.

## Installation

To install CLI-OAI, you can use the following command:

```sh
poetry add git+https://github.com/cmi-dair/cli-oai
```

## Usage

Before running CLI-OAI, make sure the environment variable `OPENAI_API_KEY` is set to your OpenAI API key.

To use the CLI, run `oai --help` in your terminal. This will display a list of available commands and their descriptions.

Here is a brief overview of the main commands:

- `oai stt <filename>`: Transcribes audio files with OpenAI's STT models. The `filename` argument is the file to transcribe. It can be any format that ffmpeg supports. Use the `--clip` option to clip the file if it is too large.

- `oai tts <text>`: Generates audio files with OpenAI's Text to Speech models. The `text` argument is the text to convert to speech.

- `oai image <prompt>`: Generates images with OpenAI's DALL-E. The `prompt` argument is the text prompt to generate the image from.

Each command has additional options that can be viewed by running `oai <command> --help`.

## Contributing

Contributions are welcome! Please see the [contributing guidelines](CONTRIBUTING.md) for more information.

## License

CLI-OAI is licensed under the terms of the [L-GPLv2.1 license](LICENSE).

## Support

If you encounter any issues or have any questions, please report them on our [issues page](https://github.com/cmi-dair/cli-oai/issues).
