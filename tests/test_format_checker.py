import pytest

from format_checker import FormatChecker


class TestFormatChecker:
    @pytest.fixture
    def function_definitions(self) -> dict:
        """Fixture for the function definitions."""
        return {
            "func_simple": {
                "parameters": {
                    "required": ["arg_type", "arg_name"],
                    "properties": {
                        "arg_type": {"type": "string"},
                        "arg_name": {"type": "string"},
                    },
                }
            },
            "func_with_numbers": {
                "parameters": {
                    "required": ["arg_type", "arg_pos"],
                    "properties": {
                        "arg_type": {
                            "type": "string",
                            "enum": ["optA", "optB", "optC"],
                        },
                        "arg_pos": {"type": "string"},
                        "arg_speed": {"type": "number", "minimum": 0, "maximum": 100},
                        "arg_level": {"type": "number", "minimum": 1, "maximum": 3},
                    },
                }
            },
        }

    def test_missing_function_calls_key(self, function_definitions: dict) -> None:
        """Test the case where 'function_calls' key is missing."""
        checker = FormatChecker(function_definitions)
        invalid_function_call = {"query": "Log random request data"}
        with pytest.raises(KeyError):
            checker._validate_function(invalid_function_call)

    def test_invalid_function_name(self, function_definitions: dict) -> None:
        """Test validation of a function call with an invalid function name."""
        checker = FormatChecker(function_definitions)
        invalid_function_call = {
            "function_calls": [
                {
                    "function_name": "invalid_function",
                    "arguments": {"arg_type": "data_x", "arg_name": "some_value"},
                }
            ]
        }
        with pytest.raises(
            KeyError, match="Function 'invalid_function' is not defined"
        ):
            checker._validate_function(invalid_function_call["function_calls"][0])

    @pytest.mark.parametrize(
        "arguments, expected_logs",
        [
            (
                {"arg_type": "optA"},
                "Missing required argument: `arg_pos` from synthesized:",
            ),
            (
                {"arg_type": "optA", "arg_pos": "position_1", "arg_speed": "fast"},
                "Argument 'arg_speed' must be of type number.",
            ),
            (
                {"arg_type": "not_in_enum", "arg_pos": "position_1"},
                "Argument 'arg_type' must be one of",
            ),
            (
                {"arg_type": "optA", "arg_pos": "position_1", "arg_speed": 150},
                "Argument 'arg_speed' is above the maximum value",
            ),
        ],
    )
    def test_argument_validation(
        self,
        function_definitions: dict,
        arguments: dict,
        expected_logs: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Parameterized test for various invalid argument cases."""
        checker = FormatChecker(function_definitions)
        invalid_function_call = {
            "function_calls": [
                {"function_name": "func_with_numbers", "arguments": arguments}
            ]
        }
        is_valid = checker._validate_function(
            invalid_function_call["function_calls"][0]
        )
        assert not is_valid
        assert any(expected_logs in record.message for record in caplog.records)

    def test_run_validation_output(self, function_definitions: dict) -> None:
        """Test the output of the `run` method for valid and invalid cases."""
        checker = FormatChecker(function_definitions)
        function_calls = [
            {
                "query": "Perform some test action with random data",
                "function_calls": [
                    {
                        "function_name": "func_simple",
                        "arguments": {"arg_type": "some_string", "arg_name": "sample"},
                    }
                ],
            },
            {
                "query": "Perform random test action on dataset",
                "function_calls": [
                    {
                        "function_name": "func_with_numbers",
                        "arguments": {
                            "arg_type": "optA",
                            "arg_pos": "position_1",
                            "arg_speed": 50,
                        },
                    }
                ],
            },
            {
                "query": "Attempt random test with out-of-range value",
                "function_calls": [
                    {
                        "function_name": "func_with_numbers",
                        "arguments": {
                            "arg_type": "optA",
                            "arg_pos": "position_1",
                            "arg_speed": 150,
                        },
                    }
                ],
            },
            {
                "query": "Attempt random test with invalid argument type",
                "function_calls": [
                    {
                        "function_name": "func_with_numbers",
                        "arguments": {
                            "arg_type": "optA",
                            "arg_pos": "position_1",
                            "arg_speed": "fast",
                        },
                    }
                ],
            },
            {
                "query": "Attempt random test with invalid enum",
                "function_calls": [
                    {
                        "function_name": "func_with_numbers",
                        "arguments": {
                            "arg_type": "not_in_enum",
                            "arg_pos": "position_1",
                        },
                    }
                ],
            },
        ]
        valid_calls, invalid_calls = checker.run(function_calls)

        assert len(valid_calls) == 2
        assert len(invalid_calls) == 3
        for invalid_call in invalid_calls:
            assert "function_name" in invalid_call["function_calls"][0]

    @pytest.mark.parametrize(
        "arguments, expected_valid",
        [
            ({"arg_type": "optA", "arg_pos": "pos", "arg_speed": 0}, True),
            ({"arg_type": "optA", "arg_pos": "pos", "arg_speed": 100}, True),
            ({"arg_type": "optA", "arg_pos": "pos", "arg_speed": -1}, False),
            ({"arg_type": "optA", "arg_pos": "pos", "arg_speed": 101}, False),
        ],
    )
    def test_numeric_boundaries(
        self, function_definitions: dict, arguments: dict, expected_valid: bool
    ):
        """Check numeric boundaries for arg_speed."""
        checker = FormatChecker(function_definitions)
        call = {
            "function_calls": [
                {
                    "function_name": "func_with_numbers",
                    "arguments": arguments,
                }
            ]
        }
        is_valid = checker._validate_function(call["function_calls"][0])
        assert is_valid == expected_valid
