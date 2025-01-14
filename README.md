# Cloai

Cloai is a generic interface to large language models. It enables the usage of
various prompting techniques (currently: chain of verification, chain of density,
instructor) across a wide variety of models, whilst using an identical interface.

## Installation

To install cloai, you can use the following command:

```sh
pip install cloai
```

## Usage

First, instantiate a client with a large language model provider. Cloai currently
supports OpenAI, Azure OpenAI, AWS Bedrock (Anthropic only), and Ollama:

#### OpenAI Client

```python
import cloai

client = cloai.OpenAiLlm(api_key="your_key", model="gpt-4o")
```

#### Ollama Client

```python
import cloai

client = cloai.OllamaLlm(
  model="llama3.2",
  base_url="http://localhost:11434/v1",
)
```

#### Azure OpenAI Client
```python
import cloai

client = cloai.AzureLlm(
  api_key="your_key",
  endpoint="your_endpoint",
  api_version="version_number",
  deployment="your_deployment"
)
```

#### AWS Bedrock (Anthropic)
```python
import cloai

client = cloai.AnthropicBedrockLlm(
  model="anthropic.claude-3-5-sonnet-20241022-v2:0",
  aws_access_key="YOUR_ACCESS_KEY",
  aws_secret_key="YOUR_SECRET_KEY",
  region="REGION",
)[llm.py](src/cloai/llm/llm.py)
```

Once your client is created, you can construct the generic interface and make use of
all the methods, regardless of which LLM you are using. Please be aware that cloai
uses asynchronous clients so you will have to await the promises. If you are in a
synchronous environment, see `asyncio.run()`.

```python
import cloai
import pydantic

model = cloai.LargeLanguageModel(client=client)

# Standard prompt
result = await model.run(system_prompt, user_prompt)

# Instructor
class Response(pydantic.BaseModel):
    is_scary: bool

result = await model.call_instructor(
  response_model=Response,
  system_prompt="Tell the user if a movie is scary.",
  user_prompt="Scary movie 3."
)

# Chain of verification
result = model.chain_of_verification(system_prompt, user_prompt)

# Chain of density
result = model.chain_of_density(text)
```



## Contributing

Contributions are welcome! Please see the [contributing guidelines](CONTRIBUTING.md) for more information.

## License

cloai is licensed under the terms of the [L-GPLv2.1 license](LICENSE).

## Support

If you encounter any issues or have any questions, please report them on our [issues page](https://github.com/childmindresearch/cloai/issues).
