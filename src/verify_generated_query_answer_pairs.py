import asyncio
import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import jsonlines

from custom_exceptions import InvalidFunctionDefinitionError
from format_checker import FormatChecker
from log_handlers import get_handlers
from semantic_checker import SemanticChecker

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
for handler in get_handlers():
    logger.addHandler(handler)


class DataVerifier:
    """
    High level class for the verification process on the generated query answer data.
    It has the following steps:
    1. Format Checker: Validate the generated data against the function definitions.
    2. Similarity Checker: Remove highly similar entries based on a threshold If enabled.
    3. Semantic Checker: Validate the query and the generated functions semantically.
    """

    def __init__(
        self,
        function_definition_path: Path,
        generated_file_path: Path,
    ) -> None:
        """
        Initialize the data verifier with the function definitions and generated data.

        Args:
            function_definitions_path (Path): Path to the function definitions file.
            generated_file_path (Path): Path to the generated data file.
        """
        # Load function definitions
        function_definition = self._load_function_definitions(function_definition_path)
        self.format_checker = FormatChecker(function_definition)
        self.semantic_checker = SemanticChecker(function_definition=function_definition)

        # Read generated data
        with jsonlines.open(generated_file_path) as reader:
            self.generated_data = list(reader)

    def _load_function_definitions(self, file_path: Path) -> dict[str, Any]:
        """
        Load and parse function definitions.

        Args:
            file_path (Path): Path to the function definitions file.

        Returns:
            dict[str, Any]: Dictionary of function definitions.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                function_definitions = json.load(f)

            parsed_functions = {}
            for func in function_definitions["functions"]:
                try:
                    function_name = func["function"]["name"]
                    if "parameters" in func["function"]:
                        parsed_functions[function_name] = {
                            "description": func["function"]["description"],
                            "parameters": func["function"]["parameters"],
                        }
                    else:
                        parsed_functions[function_name] = {
                            "description": func["function"]["description"],
                        }
                except KeyError as e:
                    raise InvalidFunctionDefinitionError(
                        f"Failed to parse function definition for {func}: {e}"
                    )
            return parsed_functions

        except Exception as e:
            logger.error(f"Failed to load function definitions from {file_path}: {e}")
            raise InvalidFunctionDefinitionError(
                f"Failed to load function definitions: {e}"
            )

    async def run(self) -> list[dict]:
        """
        Validate the generated data using several checkers.

        Returns:
            list[dict]: List of valid data.
        """
        num_total_functions = len(self.generated_data)
        logger.info(f"Number of functions to validate: {num_total_functions}")

        # Run format checker
        valid_data, format_invalid_data = self.format_checker.run(self.generated_data)
        logger.info(f"Number of functions after format checker: {len(valid_data)}")

        # Run semantic checker
        valid_data, semantic_invalid_data = await self.semantic_checker.run(valid_data)
        logger.info(
            f"Number of valid functions after semantic checker: {len(valid_data)}"
        )

        # Log complete status
        logger.info("==== Validation complete ====")
        logger.info(f"Total functions to validate: {num_total_functions}")

        logger.info(
            f"Number of invalidated functions at Format Checker: {len(format_invalid_data)}"
        )
        logger.info(
            f"Number of invalidated functions at Semantic Checker: {len(semantic_invalid_data)}"
        )
        logger.info(f"Number of valid functions: {len(valid_data)}")
        return valid_data


def parse_arguments() -> Namespace:
    """
    Parse command-line arguments.

    Returns:
        Namespace: Parsed arguments.
    """
    parser = ArgumentParser(description="Verify generated function data.")
    parser.add_argument(
        "--function-definitions-path",
        type=Path,
        required=True,
        help="Path to the function definitions JSON file.",
    )
    parser.add_argument(
        "--generated-query-answer-path",
        type=Path,
        required=True,
        help="Path to the generated query answer pairs JSONL file.",
    )
    parser.add_argument(
        "--verified-query-answer-path",
        type=Path,
        required=True,
        help="Path to the output file for verified data.",
    )
    return parser.parse_args()


def verify_generated_data(
    function_definitions_path: Path,
    generated_query_answer_path: Path,
    verified_query_answer_path: Path,
) -> None:
    """
    Perform the verification process on generated data.

    Args:
        function_definitions_path (Path): Path to the function definitions file.
        generated_query_answer_path (Path): Path to the input generated data file.
        verified_query_answer_path (Path): Path to the output verified data file.
    Returns:
        None
    """
    # Initialize the data verifier class with generated qa pairs
    data_verifier = DataVerifier(
        function_definitions_path,
        generated_query_answer_path,
    )

    # Read generated data and run the verification process
    verified_data = asyncio.run(data_verifier.run())

    # Write verified data to a file
    with jsonlines.open(verified_query_answer_path, mode="w") as writer:
        writer.write_all(verified_data)


if __name__ == "__main__":
    args = parse_arguments()
    verify_generated_data(
        function_definitions_path=args.function_definitions_path,
        generated_query_answer_path=args.generated_query_answer_path,
        verified_query_answer_path=args.verified_query_answer_path,
    )
