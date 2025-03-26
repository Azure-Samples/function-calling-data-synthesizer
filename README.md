# Data Synthesizer for Function Calling

## Overview

This repository provides a solution for generating and verifying query–function calling pairs used to fine-tune and evaluate function calling models. It comprises three main components:

- **Data Generation:** Creates synthetic query–answer pairs by leveraging seed QA examples, API function definitions, and dynamic prompt templates.
- **Multi-Stage Verification:** Uses format and semantic checkers to ensure that generated pairs are structurally correct and semantically aligned.
- **Dataset Split & Conversion:** Splits data into training, validation (and optionally test) sets and converts them to a PHI fine-tuning format.

All core processing logic is contained in the **src** directory, which includes modules for generation, verification, format conversion, logging, inference, and custom exception handling.
Data Generation and Multi-Stage Verification components design are inspired by [APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets](https://arxiv.org/pdf/2406.18518). This solution automates the creation of diverse query-function pairs, significantly reducing the need for manual annotation

## Architecture

The system implements a data synthesis pipeline (`pipelines/main.yaml`) with three core modules:

### 1. Data Generation

- Uses seeded QA examples and API function definitions to generate new synthetic query–function calling pairs.
- The module `src/generate_synthetic_pairs.py` handles prompt creation and asynchronous model calls.

### 2. Multi-Stage Verification

- **Format Checker:** (`src/format_checker.py`) Validates required arguments, data types, enum values, numeric boundaries, and conditional requirements.
- **Semantic Checker:** (`src/semantic_checker.py`) Uses a separate prompt and asynchronous inference to ensure that the generated calls semantically align with the query.

### 3. Data Splitting and Conversion

- **Data Split:** (`src/split_data.py`) Splits verified data into training and validation sets using stratified sampling.
- **Chat Message Format Conversion:** (`src/apply_chat_message_format.py`) Converts the split data into a format suitable for fine-tuning models. This conversion wraps each data point into a message sequence that includes:
  - A system prompt containing instructions.
  - The user query.
  - The assistant’s (converted) function call formatted as a JSON string.

#### Example of Converted Chat Message Format for Fine Tuning

Below is a sample output after conversion (one-line in JSONL file):

````json
{
  "messages": [
    { "role": "system", "content": "You are an in-car assistant with a list of tools. ..." },
    { "role": "user", "content": "Play the song 'Starlight' on MusicBox" },
    { "role": "assistant", "content": "[{\"function_name\": \"play_audio_track\", \"arguments\": {\"service\": \"MusicBox\", \"media_type\": \"track\", \"title\": \"Starlight\"}}]" }
  ]
}
````

## License Considerations for Data Synthesis for Fine Tuning

> [!WARNING]
> A lot of LLMs typically adopt restrictive licenses for synthesizing data and using them for fine-tuning another SLM/LLM. For example, as of Mar 6, 2025, [Azure OpenAI Service Specific Terms](https://www.microsoft.com/licensing/terms/productoffering/MicrosoftAzure/EAEAS#ServiceSpecificTerms:~:text=under%20the%20circumstances.-,Azure%20OpenAI%20Service,-In%20addition%20to) doesn't allow that usage except for certain permitted cases in the terms. Check the terms first if you intend to use Azure OpenAI for data synthesis for fine tuning. This solution is heavily tested with [**Phi3.5 MoE instruct**](https://learn.microsoft.com/en-us/azure/ai-foundry/model-inference/concepts/models#microsoft) as it is one of the most capable models that still adopts permissive, MIT license. Check [available models on Azure AI Foundry](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/model-catalog-overview#available-models-for-supported-deployment-options) and check license on [Azure AI model catalog](https://azure.microsoft.com/en-us/products/ai-model-catalog) first and decide what model has acceptable license conditions for your team.

## Getting Started

### Prerequisites

- [Visual Studio Code](https://code.visualstudio.com/) installed
- [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) installed
- Docker ([Installation Guide](https://docs.docker.com/get-docker/))

### Setup & Local Development

1. Clone the repository

    ```bash
    git clone git@github.com:cse-labs/function-calling-data-synthesizer.git
    cd function-calling-data-synthesizer
    code .
    ```

2. Copy `.env.example` and rename it to `.env`. This file will be automatically loaded into the dev container as environment variables.

   **Important:** If you skip this step, you will get an error when trying to build the container.

3. Deploy Azure Machine Learning Workspace and Azure AI Foundry
4. Fill out `WORKSPACE_NAME` and `RESOURCE_GROUP` in `.env` with your Azure Machine Learning Workspace settings
5. Use the Dev Containers: Reopen in Container command from the Command Palette (F1, ⇧⌘P) and open it in a Dev Container
6. Deploy Phi3.5 MoE instruct (the model type we tested this solution with mainly) or [any available models on Azure AI model catalog](https://learn.microsoft.com/en-us/azure/ai-foundry/model-inference/concepts/models) on Azure AI foundry
7. Get deployed model(s) API endpoints and API keys from your deployed model(s) and fill `MODEL_API_BASE_URL[1-]` and `MODEL_API_KEY[1-]` with those values. Currently `inference.py` and `.env.example` support only two endpoints but you can add more endpoints if you face token limit. Most models on Azure AI Foundry model catalog don't allow increasing token limit as of Mar 6th, 2025 without going through a support request. Deploying more endpoints is one quick way to address token limit errors.
8. Update prompts adopted in `prompts/generator_config.yaml` and `prompts/system_prompt.txt` based on your use case. Current example prompt templates show Car AI use case version of prompts.

## Dataset Preparation

You need two kinds of input data to be prepared:

- **Function definitions JSON file**:
  OpenAI's Function Spec is supported. Follow [OpenAI's API reference](https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools) and [function calling guide](https://platform.openai.com/docs/guides/function-calling#defining-functions) to define your own function definitions. An example is located at `data/functions_definition.json`.

- **Seed QA (Query and function calling Answer) Dataset JSONL file**:
  This dataset serves as seed few-shot examples for generating new synthetic query-answer pairs. This dataset should be in JSONL format, where every line is a JSON object containing the keys `"query"` and `"function_calls"`. You can create your own seed QA dataset or use the provided evaluation dataset example located at `data/examples/seed_qa_dataset.jsonl`.

## Directory Structure

This section gives you overview of the directory structure of this solution. Only essential files are covered in this structure graph for simplicity. The directory structure is as follows:

```
├── .devcontainer/        # Dockerfile and dev container configuration
├── .github/              # CI pipeline for Github Actions
├── data/                 # Data directory
│   └── examples/         # Example function definition and seed QA data
├── pipelines/            # Pipeline files for AML CLI
├── prompts/              # Prompt files for data synthesis
├── src/                  # Source code for data synthesis
├── tests/                # Unit and functional tests
├── .env.example          # Environment variable examples
├── .amlignore            # Files ignored when uploading to AML pipeline
├── .pre-commit-config.yaml # Pre-commit configuration
├── pyproject.toml        # Project python dependencies + python tool configurations
└── README.md             # This file
```

### Overview of the src Directory

The `src` directory contains:

- **Generation and Inference Modules:** (`generate_synthetic_pairs.py`, `inference.py`) for data synthesis via model inference.
- **Verification Modules:** (`format_checker.py` and `semantic_checker.py`) that perform multi-stage validation.
- **Data Conversion & Splitting Modules:** (`apply_chat_message_format.py`, `split_data.py`) to prepare data for training.
- **Utility Modules:** (e.g., `log_handlers.py`, `custom_exceptions.py`) used throughout the pipeline.

## Running Scripts Locally

### Data Generation

Run the following command:

```bash
python src/generate_synthetic_pairs.py \
 --config-path prompts/generator_config.yaml \
 --qa-jsonl-path data/YOUR_SEED_QA_DATASET.jsonl \
 --function-definitions-path data/YOUR_LOCAL_FUNCTION_DEFINITION.json \
 --output-path data/YOUR_OUTPUT_PATH.jsonl
```

### Data Verification

```bash
python src/verify_generated_query_answer_pairs.py \
  --generated-query-answer-path data/YOUR_GENERATED_QA_INPUT_PATH.jsonl \
  --function-definitions-path data/YOUR_LOCAL_FUNCTION_DEFINITION.json \
  --verified-query-answer-path data/YOUR_OUTPUT_PATH.jsonl
```

### Data Splitting and Chat Message Format Conversion for Fine Tuning

```bash
python src/split_data.py \
  --input-file-path data/YOUR_VERIFIED_DATA.jsonl \
  --train-output-path data/train.jsonl \
  --val-output-path data/validation.jsonl \
  --test-size 0.3

python src/apply_chat_message_format.py \
  --input-path data/train.jsonl \
  --output-path data/fine_tuning_format_conversion.jsonl
```

## Running Data Synthesis on Azure ML

This repository includes a complete AML pipeline (see `pipelines/main.yaml`). The pipeline runs all steps—from data generation, verification, splitting, to Finetuning format conversion in sequence.

### Prerequisites

1. Create AML custom environment using the Dockerfile:

```sh
az login
az ml environment create --file pipelines/create_aml_custom_env.yaml --workspace-name $WORKSPACE_NAME --resource-group $RESOURCE_GROUP
```

2. Register local datasets as Azure Machine Learning Data Assets with the following commands:

```sh
az login
az ml data create --file pipelines/register_functions_definition.yaml -w $WORKSPACE_NAME -g $RESOURCE_GROUP
az ml data create --file pipelines/register_seed_qa_dataset.yaml -w $WORKSPACE_NAME -g $RESOURCE_GROUP
```

### Submitting a data synthesis pipeline job

To submit a job:

```bash
az ml job create --file pipelines/main.yaml --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME
```

For further details, refer to the [Azure ML documentation](https://learn.microsoft.com/azure/machine-learning/).

## Adding a new python package

We keep the necessary libraries for this repository in `pyproject.toml`

```bash
poetry add package_name
```

## Running Tests

To run the unit tests, execute:

```bash
PYTHONPATH=src pytest tests/
```

Ensure all tests pass before submitting changes.

## Limitations

This solution currently assumes that each query has one intent solved by a single function call. Changing this may require an update to the inference and verification logic

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us

the rights to use your contribution. For details, visit <https://cla.opensource.microsoft.com>.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
