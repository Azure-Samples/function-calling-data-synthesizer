from __future__ import annotations

import asyncio
import json
import logging
import random
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import jsonlines
import tqdm
import tqdm.asyncio
import yaml
from azure.ai.inference.models import (
    UserMessage,
)
from dotenv import load_dotenv

from inference import async_inference
from log_handlers import get_handlers

load_dotenv()


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for handler in get_handlers():
    logger.addHandler(handler)


@dataclass
class ErrorCounts:
    no_json_section: int = 0
    json_decode_error: int = 0
    missing_keys: int = 0

    def __iadd__(self, other: ErrorCounts) -> ErrorCounts:
        self.no_json_section += other.no_json_section
        self.json_decode_error += other.json_decode_error
        self.missing_keys += other.missing_keys
        return self


class QASampler:
    """
    A class to load and sample question-answer pairs from a JSONL file.
    Attributes:
    -----------
    path : Path
        The path to the JSONL file containing the question-answer pairs.
    per_function_annotations : dict[str, list]
        A dictionary where the keys are function names and the values are lists of question-answer pairs.
    Methods:
    --------
    sample(class_name: str, n: int) -> list[dict[str, Any]]:
        Samples `n` question-answer pairs for a given function name.
    """

    def __init__(self, path: Path):
        """
        Loads data from a JSON Lines file specified by `self.path` and populates the `self.per_function_annotations` dictionary.
        The method assumes that each query in the JSON Lines file only calls one function. It reads the file line by line,
        extracts the function name from the first function call in each query, and appends the entire query object to the list
        of annotations for that function name in `self.per_function_annotations`.
        Raises:
            FileNotFoundError: If the file specified by `self.path` does not exist.
        """
        self.path = path
        self.per_function_annotations: dict[str, list] = defaultdict(list)
        with jsonlines.open(self.path) as reader:
            for obj in reader:
                # currently we assume each query only calls one function so grab the first one only for classification
                self.per_function_annotations[
                    obj["function_calls"][0]["function_name"]
                ].append(obj)

    def sample(self, class_name: str, n: int) -> list[dict[str, Any]]:
        """
        Samples a specified number of annotations for a given class.
        When the number of annotations requested is greater than the number of annotations available for the class,
        the method logs a warning and samples all available annotations.

        Args:
            class_name (str): The name of the class to sample annotations from.
            n (int): The number of annotations to sample.
        Returns:
            list[dict[str, Any]]: A list of sampled annotations.
        """
        if (max_annoation_count := len(self.per_function_annotations[class_name])) < n:
            logger.warning(
                f"Wanted to sample {n} cases but only {max_annoation_count} annotations exists for {class_name}. "
                f"Sampling all annotations ({max_annoation_count} cases) for {class_name}"
            )
            sampled_qa = random.sample(
                self.per_function_annotations[class_name], max_annoation_count
            )
            return sampled_qa
        else:
            sampled_qa = random.sample(self.per_function_annotations[class_name], n)
            return sampled_qa

    def get_classes(self) -> list[str]:
        """
        Returns the list of classes in per_function_annotations.
        Returns:
            list[str]: The list of classes in per_function_annotations.
        """
        return list(self.per_function_annotations.keys())


class APISampler:
    def __init__(self, path: Path):
        """
        Loads data from a JSON Lines file specified by `self.path` and populates the `self.per_function_annotations` dictionary.
        The method assumes that each query in the JSON Lines file only calls one function. It reads the file line by line,
        extracts the function name from the first function call in each query, and appends the entire query object to the list
        of annotations for that function name in `self.per_function_annotations`.
        Raises:
            FileNotFoundError: If the file specified by `self.path` does not exist.
        """
        self.path = path
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.function_definitions = {
                func["function"]["name"]: func["function"] for func in data["functions"]
            }

    def get_classes(self) -> list[str]:
        """
        Returns the list of classes in per_function_annotations.
        Returns:
            list[str]: The list of classes in per_function_annotations.
        """
        return list(self.function_definitions.keys())


@dataclass
class SyntheticDataGeneratorConfig:
    """Configuration for generating synthetic data for a function calling model."""

    template: str
    """The template for the user message passed to the generating model. The template should
    contain named placeholders that get replaced before calling the generation model.
    The following placeholders are supported:
    - {{examples}}: a few shot examples for the query - function calling pairs
    - {{function_name}}: The name of the function to call.
    - {{function_description}}: The description of the function.
    - {{number}}: the number of pairs to generate
    - {{persona_instruction}}: the persona to be applied for generation. Only used when personality.enabled is true in config
    """
    personality: dict[str, bool | str | dict[str, list[str]]]
    """
    This field is used to add persona to the generation.
    """
    model_configuration: dict
    """The configuration to pass to the generating model. This must include the "model".
    The values get passed to an ChatCompletionsClient for generating a chat completion"""
    generations_per_function: int = 10
    """The number of synthetic data samples to generate for each function. Remember that the model
    can generate multiple user messages per call, so the total number of data points will be higher"""
    few_shot_samples: int = 5
    """The number of few shot examples to sample for each generation."""

    @staticmethod
    def from_yaml(file_path: Union[str, Path]) -> SyntheticDataGeneratorConfig:
        """
        Load the configuration from a YAML file.
        Args:
            file_path (Union[str, Path]): Generator config file path
         Returns:
            SyntheticDataGeneratorConfig: The configurations with file content
        """

        with open(file_path, "r") as config_file:
            config_data = yaml.safe_load(config_file)

        return SyntheticDataGeneratorConfig(
            template=config_data["template"],
            personality=config_data["personality"],
            model_configuration=config_data["model_configuration"],
            generations_per_function=config_data.get("generations_per_function", 10),
            few_shot_samples=config_data.get("few_shot_samples", 5),
        )


class SyntheticDataGenerator:
    """
    SyntheticDataGenerator is a class responsible for generating synthetic data based on function definitions and QA samples. It uses a model client to generate responses and extracts data points from these responses.
    Attributes:
        config (SyntheticDataGeneratorConfig): Configuration for the synthetic data generation process.
        phi_client (ChatCompletionsClient): Client for interacting with the Phi model.
        qa_sampler (QASampler): Sampler for QA examples.
        api_sampler (APISampler): Sampler for API function definitions.
    Methods:
        __init__(config: SyntheticDataGeneratorConfig, qa_jsonl_path: Path, function_definitions_path: Path):
            Initializes the SyntheticDataGenerator with the provided configuration, QA samples path, and function definitions path.
        generate_data() -> list[dict[str, Any]]:
            Generates synthetic data by iterating over function definitions and using a Phi model to generate responses.
    """

    def __init__(
        self,
        config: SyntheticDataGeneratorConfig,
        qa_jsonl_path: Path,
        function_definitions_path: Path,
    ):
        self.config = config
        self.qa_sampler = QASampler(qa_jsonl_path)
        self.api_sampler = APISampler(function_definitions_path)
        if set(self.api_sampler.get_classes()) != set(self.qa_sampler.get_classes()):
            raise ValueError(
                f"API and QA samplers should have the same classes. API classes: {self.api_sampler.get_classes()}, QA classes: {self.qa_sampler.get_classes()}"
            )

    def generate_data(self) -> list[dict[str, Any]]:
        """
        Generates synthetic data by iterating over function definitions and using a phi model to generate responses.
        For each function in the API sampler's function definitions, this method:
        - Samples examples using the QA sampler.
        - Randomly determines the number of data points to generate.
        - Formats a user message based on a template, replacing variables with actual values.
        - Sends the message to the model client to generate a response.
        - Extracts data points from the model's response.
        - Drops any duplicated data from the extracted data points based on queries.
        - Accumulates the extracted data points into a list.
        Returns:
            list[dict[str, Any]]: A list of dictionaries representing the generated synthetic data points.
        """

        total_generated_data: list[dict[str, Any]] = []
        total_error_counts = ErrorCounts()
        logger.info(
            f"Generating synthetic data for {len(self.api_sampler.get_classes())} functions"
        )
        for (
            function_name,
            function_definition,
        ) in tqdm.tqdm(self.api_sampler.function_definitions.items()):
            logger.info(f"Target function: {function_name}")
            logger.info(f"Used function definition: {function_definition}")
            generated_data, error_counts = asyncio.run(
                self._async_generate_multiple(
                    function_name=function_name,
                    function_definition=function_definition,
                    n=self.config.generations_per_function,
                )
            )
            total_generated_data.extend(generated_data)
            total_error_counts += error_counts

        # when we generate data, we might have duplicates. We need to remove them
        # currently we just keep only the first one regardless of inferred function callings
        used_queries = set()
        deduped_data = []
        deduped_count = 0
        for data_point in total_generated_data:
            query = data_point["query"]
            if query in used_queries:
                logger.warning(f"Duplicated query: {query}")
                deduped_count += 1
            else:
                used_queries.add(query)
                deduped_data.append(data_point)

        logger.info(
            f"No JSON section error counts: {total_error_counts.no_json_section}"
        )
        logger.info(f"JSON decode error counts: {total_error_counts.json_decode_error}")
        logger.info(f"Missing keys error counts: {total_error_counts.missing_keys}")
        logger.info(f"Deduped {deduped_count} data points")
        logger.info(f"Generated {len(deduped_data)} data points in total")

        return deduped_data

    def _prepare_user_message(
        self, function_name: str, function_definition: dict[str, Any], gen_count: int
    ) -> str:
        """
        Generate a user message by populating a template with provided function details, examples, and additional personality attributes if enabled.
        Parameters:
            function_name (str): The name of the function for which the message is generated.
            function_definition (dict[str, Any]): A dictionary containing the function's definition or description.
            gen_count (int): A numerical identifier used within the message, representing the number of samples to generate.
        Returns:
            str: The formatted user message with all placeholders replaced by actual values.
        """

        examples = self.qa_sampler.sample(function_name, self.config.few_shot_samples)
        variables = {
            "{{function_name}}": function_name,
            "{{function_description}}": json.dumps(function_definition),
            "{{number}}": str(gen_count),
            "{{examples}}": "\n".join([json.dumps(example) for example in examples]),
        }

        user_message = self.config.template
        for key, value in variables.items():
            user_message = user_message.replace(key, value)

        if self.config.personality["enabled"]:
            persona_instruction = self.config.personality["template"]
            for field_name, field_values in self.config.personality["values"].items():
                persona_instruction = persona_instruction.replace(
                    f"{{{{{field_name}}}}}", random.choice(field_values)
                )
            user_message = user_message.replace(
                "{{persona_instruction}}", persona_instruction
            )
        else:
            user_message = user_message.replace("{{persona_instruction}}", "")

        logger.debug(f"Final user message: {user_message}")

        return user_message

    async def _generate_for_function(
        self, function_name: str, function_definition: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], ErrorCounts] | None:
        """
        Asynchronously generates synthetic input data for a given function.
        Args:
            function_name (str): The name of the function for which to generate data.
            function_definition (dict[str, Any]): The definition of the function, including its parameters and other metadata.
        Returns:
            tuple[list[dict[str, Any]] | ErrorCounts] | None: A tuple containing:
                - A list of dictionaries representing the generated data points
                - An ErrorCounts object representing the count of errors encountered during data extraction.
             or None if generation failed.
        Raises:
            Exception: If the asynchronous inference call fails.
        Notes:
            - The function uses a template to format the user message and sends it to an asynchronous inference model.
            - The number of data points to generate is randomly determined between 1 and 5.
            - The function logs the generated user message and any errors encountered during the process.
            - If the number of generated data points does not match the expected count, a log message is recorded.
        """

        total_error_counts = ErrorCounts()
        gen_count = random.randint(1, 5)
        user_message = self._prepare_user_message(
            function_name=function_name,
            function_definition=function_definition,
            gen_count=gen_count,
        )

        logger.debug(f"{function_name}: {user_message}")

        messages = [UserMessage(content=user_message)]
        try:
            response = await async_inference(
                messages=messages,
                model_configuration=self.config.model_configuration,
            )
        except Exception as e:
            # We need to make sure we do not loose data if the model fails
            logger.warning(f"Failed to generate data for {function_name}: {e}")
            return None

        message_content = response.choices[0].message.content  # type: ignore
        extracted_data_points, error_counts = _extract_data(message_content)
        total_error_counts += error_counts

        if gen_count != len(extracted_data_points):
            logger.info(
                f"Generated {len(extracted_data_points)} data points instead of {gen_count}"
            )
        else:
            logger.info(f"Generated {len(extracted_data_points)} data points")
        return extracted_data_points, error_counts

    async def _async_generate_multiple(
        self, function_name: str, function_definition: dict[str, Any], n: int
    ) -> tuple[list[dict[str, Any]], ErrorCounts]:
        """
        Asynchronously generate multiple sets of synthetic inputs for a given function.
        Args:
            function_name (str): The name of the function for which to generate inputs.
            function_definition (dict[str, Any]): The definition of the function, including its parameters and types.
            n (int): The number of sets of inputs to generate.
        Returns:
            tuple[list[dict[str, Any]], ErrorCounts]: A tuple containing a list of dictionaries with the generated inputs
            and an ErrorCounts object summarizing any errors encountered during generation.
        """

        all_extracted_data_points = []
        total_error_counts = ErrorCounts()
        tasks = [
            self._generate_for_function(
                function_name=function_name, function_definition=function_definition
            )
            for _ in range(n)
        ]
        results = await tqdm.asyncio.tqdm.gather(*tasks)
        for result in results:
            if result is not None:
                extracted_data_points, error_counts = result
                all_extracted_data_points.extend(extracted_data_points)
                total_error_counts += error_counts
        return all_extracted_data_points, total_error_counts


def _extract_data(message_content: str) -> tuple[list[dict[str, Any]], ErrorCounts]:
    """
    Extracts JSON data from a given message content string.
    This function processes each line of the input string, attempting to extract and parse JSON objects.
    It collects valid JSON objects that contain both "query" and "function_calls" keys into a list.
    It also tracks and logs various errors encountered during the extraction process.
    Args:
        message_content (str): The input string containing lines of text with embedded JSON objects.
    Returns:
        tuple[list[dict[str, Any]], ErrorCounts]: A tuple containing:
            - A list of dictionaries representing the successfully parsed JSON objects.
            - An ErrorCounts object tracking the number of different types of errors encountered.
    """
    lines = message_content.strip().strip("`").replace("\n\n", "\n").split("\n")
    cleaned = []
    error_counts = ErrorCounts()
    for line in lines:
        # Disregard everything before the first {
        start = line.find("{")
        if start == -1:
            error_counts.no_json_section += 1
            logger.warning(f"No {{ in generation: {line}")
            continue
        # Find the last }
        end = line.rfind("}")
        if end == -1:
            error_counts.no_json_section += 1
            logger.warning(f"No }} in generation: {line}")
            continue
        json_section = line[start : end + 1]
        try:
            parsed = json.loads(json_section)
        except json.JSONDecodeError:
            error_counts.json_decode_error += 1
            logger.warning(f"Failed to decode JSON: {json_section}")
            continue
        if "query" in parsed and "function_calls" in parsed:
            cleaned.append(parsed)
        else:
            error_counts.missing_keys += 1
            logger.warning(f'"query" or "function_calls" key is missing: {parsed}')
    return cleaned, error_counts


def generate_synthetic_data(
    config_path: Path,
    qa_jsonl_path: Path,
    function_definitions_path: Path,
    output_path: Path,
) -> None:
    """
    Generates synthetic data based on the provided configuration and input files.
    Args:
        config_path (Path): Path to the YAML configuration file.
        qa_jsonl_path (Path): Path to the QA JSONL file containing question-answer pairs.
        function_definitions_path (Path): Path to the file containing function definitions.
        output_path (Path): Path where the generated synthetic data will be saved.
    Raises:
        FileNotFoundError: If any of the specified files do not exist.
    Returns:
        None
    """

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    if not qa_jsonl_path.exists():
        raise FileNotFoundError(f"QA jsonl file not found at {qa_jsonl_path}")
    if not function_definitions_path.exists():
        raise FileNotFoundError(
            f"Function definitions file not found at {function_definitions_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = SyntheticDataGeneratorConfig.from_yaml(config_path)
    generator = SyntheticDataGenerator(
        config,
        qa_jsonl_path=qa_jsonl_path,
        function_definitions_path=function_definitions_path,
    )
    generated_data = generator.generate_data()

    with jsonlines.open(output_path, "w") as f:
        f.write_all(generated_data)


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=Path,
        help="The path to the configuration yaml file to use for generating the synthetic data.",
    )
    parser.add_argument(
        "--qa-jsonl-path",
        type=Path,
        help="The path to QA pair jsonl file to use for generating the synthetic data.",
    )
    parser.add_argument(
        "--function-definitions-path",
        type=Path,
        default="functions.json",
        help="The path to function definitions json file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default="synthetic_data.jsonl",
        help="The output file to save the synthetic data to.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    generate_synthetic_data(
        config_path=args.config_path,
        qa_jsonl_path=args.qa_jsonl_path,
        function_definitions_path=args.function_definitions_path,
        output_path=args.output_path,
    )
