import logging
from typing import Any, Tuple

from tqdm import tqdm

from log_handlers import get_handlers

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
for handler in get_handlers():
    logger.addHandler(handler)


class FormatChecker:
    """
    Format checker is responsible for validating the generated data against the function definitions.
    It checks the following:
    1. Required arguments for the function
    2. Argument types
    3. Enum values
    4. Numeric ranges
    5. Conditional requirements
    """

    def __init__(self, function_definition: dict[str, Any]):
        """
        Initialize the FormatChecker object.
        Requires the function definitions for validation.

        Args:
            function_definition (dict[str, Any]): Function definitions.
        """
        self.functions = function_definition

    def _check_func_definition_has_parameters(self, function_name: str) -> bool:
        """
        Check if the function has defined parameters.

        Args:
            function_name (str): Name of the function.

        Returns:
            bool: Boolean indicating if the function has defined parameters.

        Raises:
            KeyError: If the function is not defined in self.functions.
        """
        if function_name not in self.functions:
            raise KeyError(f"Function '{function_name}' is not defined.")
        return "parameters" in self.functions[function_name]

    def _extract_function_call_details(
        self, function_call: dict[str, Any]
    ) -> Tuple[str, dict[str, Any], dict[str, Any]]:
        """
        Extract function name, arguments, and properties from the function call.

        Args:
            function_call (dict[str, Any]): Function call details.

        Returns:
            Tuple[str, dict[str, Any], dict[str, Any]]: Function name, arguments, and properties.
        """
        # Extract function name and arguments
        function_name = function_call["function_name"]
        if function_name not in self.functions:
            raise KeyError(f"Function '{function_name}' is not defined.")
        arguments = function_call["arguments"]
        # Extract expected properties for the function
        properties = self.functions[function_name]["parameters"].get("properties", {})
        if not properties:
            raise KeyError(f"Properties are not defined for function '{function_name}'")
        return function_name, arguments, properties

    def _validate_required_arguments(self, function_call: dict[str, Any]) -> bool:
        """
        Validate if all required arguments are present in the function call.

        Args:
            function_call (dict[str, Any]): Function call details.

        Returns:
            bool: Boolean indicating if the function call is valid.
        """
        # Extract function name and arguments
        function_name, arguments, _ = self._extract_function_call_details(function_call)
        required_arguments = self.functions[function_name]["parameters"].get(
            "required", []
        )

        # Check if all required arguments are present
        for required_argument in required_arguments:
            if required_argument not in arguments:
                logger.warning(
                    f"Missing required argument: `{required_argument}` from synthesized: {function_call}"
                )
                return False
        return True

    def _validate_argument_types(self, function_call: dict[str, Any]) -> bool:
        """
        Validate the argument types in the function call.

        Args:
            function_call (dict[str, Any]): Function call details.

        Returns:
            bool: Boolean indicating if the function call is valid.
        """
        # Extract function arguments
        _, arguments, properties = self._extract_function_call_details(function_call)

        # Check if the argument types are correct
        for argument, value in arguments.items():
            if argument in properties:
                expected_argument_type = properties[argument].get("type")
                if not expected_argument_type:
                    logger.warning(
                        f"Type is not defined for argument'{argument}' for function '{function_call['function_name']}'."
                    )
                    return False

                # Define expected argument types for mapping
                arg_type = {
                    "string": str,
                    "number": (int, float),
                    "boolean": bool,
                    "array": list,
                    "object": dict,
                }.get(expected_argument_type)

                # Validate argument type
                if arg_type:
                    if isinstance(arg_type, tuple):
                        if not issubclass(type(value), arg_type):
                            logger.warning(
                                f"Argument '{argument}' must be of type {expected_argument_type}. Got type: {type(value).__name__} with value: {value}"
                            )
                            return False
                    else:
                        if type(value) is not arg_type:
                            logger.warning(
                                f"Argument '{argument}' must be of type {type(arg_type)}. Got type: {type(value).__name__} with value: {value}"
                            )
                            return False
        return True

    def _validate_enums(self, function_call: dict[str, Any]) -> bool:
        """
        Validate if the argument values are part of the enum values specified in the function definition.

        Args:
            function_call (dict[str, Any]): Function call details.

        Returns:
            bool: Boolean indicating if the function call is valid.
        """
        # Extract function arguments and expected properties
        _, arguments, properties = self._extract_function_call_details(function_call)

        # Check if the argument values are part of the enum values
        for argument, value in arguments.items():
            if argument in properties:
                enum_values = properties[argument].get("enum")
                if enum_values and value not in enum_values:
                    logger.warning(
                        f"Argument '{argument}' must be one of {enum_values}. Got: {value}"
                    )
                    return False
        return True

    def _validate_numeric_ranges(self, function_call: dict[str, Any]) -> bool:
        """
        Validate if the numeric argument values are within the specified range.

        Args:
            function_call (dict[str, Any]): Function call details.

        Returns:
            bool: Boolean indicating if the function call is valid.
        """
        # Extract function arguments and expected properties
        _, arguments, properties = self._extract_function_call_details(function_call)

        # Check if the numeric argument values are within the specified range
        for argument, value in arguments.items():
            prop = properties.get(argument)
            if not prop or prop.get("type") != "number":
                continue

            if not isinstance(value, (int, float)):
                logger.warning(f"Argument '{argument}' must be a number. Got: {value}")
                return False

            # Validate numeric ranges
            if "minimum" in prop and arguments[argument] < prop["minimum"]:
                logger.warning(
                    f"Argument '{argument}' is below the minimum value: {prop['minimum']}"
                )
                return False
            if "maximum" in prop and arguments[argument] > prop["maximum"]:
                logger.warning(
                    f"Argument '{argument}' is above the maximum value: {prop['maximum']}"
                )
                return False

        return True

    def _validate_conditional_requirements(self, function_call: dict[str, Any]) -> bool:
        """
        Validate if the conditional requirements are met for the function call.

        Args:
            function_call (dict[str, Any]): Function call details.

        Returns:
            bool: Boolean indicating if the function call is valid.
        """
        function_name, arguments, _ = self._extract_function_call_details(function_call)

        # NOTE: Add conditional requirements for each function here
        # such as if argument A is present, then argument B is required
        # depending on your function definitions, you may not need this check

        # example:
        # if function_name == "func1":
        #     if "action" in arguments and "app_name" not in arguments:
        #         logger.warning(
        #             f"Argument 'app_name' is required when 'action' is present for function '{function_name}'"
        # return False
        return True

    def _validate_function(self, function_call: dict[str, Any]) -> bool:
        """
        Validate the function call with all validators
        For each function call, validate the required arguments, argument types, enums, numeric ranges, and conditional requirements.
        If any of the validators fail, return False.
        When defined functions do not have arguments, validate if the arguments are empty.

        Args:
            function_call (dict[str, Any]): One function call to validate

        Returns:
            bool: Boolean indicating if the function call is valid.
        """

        if self._check_func_definition_has_parameters(function_call["function_name"]):
            validators = [
                self._validate_required_arguments,
                self._validate_argument_types,
                self._validate_enums,
                self._validate_numeric_ranges,
                self._validate_conditional_requirements,
            ]
            for validator in validators:
                is_valid = validator(function_call)
                if not is_valid:
                    return False
            return True
        else:
            logger.info(
                f"function: {function_call['function_name']} does not have defined parameters. Checking if inferred arguments are empty."
            )
            # some defined functions do not have arguments and for those, we need to validate if the arguments are empty
            return function_call["arguments"] == {}

    def run(
        self, function_calls: list[dict[str, Any]]
    ) -> Tuple[list[dict], list[dict]]:
        """
        Run the validation for all function calls.

        Args:
            function_calls (list[dict[str, Any]]): List of function calls to validate.

        Returns:
            Tuple[list[dict], list[dict]]: Tuple containing the list of valid and invalid function calls.
        """
        valid_calls, invalid_calls = [], []

        logger.info(f"Number of functions to validate: {len(function_calls)}")

        for function_call in tqdm(
            function_calls, desc="Validating function calls for format"
        ):
            try:
                function_call_details = function_call.get("function_calls", [{}])[0]
                is_valid = self._validate_function(function_call_details)
                if is_valid:
                    valid_calls.append(function_call)
                else:
                    invalid_calls.append(function_call)
            except Exception as err:
                logger.warning(f"Malformed function call: {err}")
                invalid_calls.append(function_call)
        return valid_calls, invalid_calls
