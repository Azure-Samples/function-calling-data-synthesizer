import logging
from typing import Any

import yaml
from azure.ai.inference.models import UserMessage
from tqdm.asyncio import tqdm

from inference import async_inference
from log_handlers import get_handlers

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
for handler in get_handlers():
    logger.addHandler(handler)


class SemanticChecker:
    """A class to perform semantic checks on user queries and function calls.

    This class validates the semantic correctness of function calls generated
    by another model against the original user queries.
    """

    def __init__(
        self,
        config_path: str = "prompts/generator_config.yaml",
        function_definition: dict[str, Any] = {},
    ):
        """Initialize the SemanticChecker with the given configuration file and function definition.

        Args:
            config_path (str): Path to the configuration file. Defaults to "prompts/generator_config.yaml".
            function_definition (dict): Definition of the function to be validated
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            self.model_config = config["model_configuration"]
        self.function_definition = function_definition

    async def run(
        self, function_calls_to_semantic_check: list[dict[str, Any]]
    ) -> tuple[list[dict], list[dict]]:
        """Validates whether multiple function calls match their corresponding queries.

        Args:
            function_calls_to_semantic_check (list[dict]): List of dictionaries containing:
                - query (str): The user query
                - function_calls (list[dict]): The function calls to be validated
        Returns:
            tuple[list[dict], list[dict]]: A tuple containing:
                - List of results that passed semantic validation
                - List of results that failed semantic validation
        """
        # Extract queries and function calls
        queries = [call["query"] for call in function_calls_to_semantic_check]

        # Extract only the first function call for each query
        function_calls = [
            func_call["function_calls"][0]
            for func_call in function_calls_to_semantic_check
        ]
        with open("prompts/semantic_checker.txt", "r") as f:
            validation_prompt_template = f.read()
        messages = [
            UserMessage(
                content=validation_prompt_template.format(
                    func_desc=str(self.function_definition),
                    query=query,
                    func_call=func_call,
                )
            )
            for query, func_call in zip(queries, function_calls)
        ]
        validation_config = self.model_config.copy()
        # set higher max_tokens for semantic checker so that thought part can be well captured
        validation_config["max_tokens"] = 1000
        # set temperature to 0 to get deterministic judgement
        validation_config["temperature"] = 0

        tasks = [
            async_inference(
                messages=[message],
                model_configuration=validation_config,
            )
            for message in messages
        ]
        try:
            # Process all validations in parallel
            responses = await tqdm.gather(
                *tasks, desc="Validating function calls for semantic"
            )
        except Exception as e:
            logger.warning(f"An error occurred during semantic check: {str(e)}")
            return [], []

        valid_results = []
        invalid_results = []

        for query, response, func_call in zip(queries, responses, function_calls):
            result_content = response.choices[0].message.content.strip()
            result = {
                "query": query,
                "function_calls": [func_call],
            }
            is_valid = "yes" in result_content.lower()

            if is_valid:
                valid_results.append(result)
            else:
                invalid_results.append(result)
                logger.warning(
                    f"Semantic Check failed for query: {query}\nFunction calling: {func_call}\nValidation Reason: {result_content}"
                )

        logger.info(
            f"Semantic Check Complete - Valid: {len(valid_results)}, Invalid: {len(invalid_results)}"
        )
        return valid_results, invalid_results
