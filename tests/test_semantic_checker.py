import pytest
from azure.ai.inference.models import ChatCompletions

from semantic_checker import SemanticChecker


@pytest.fixture
def semantic_checker():
    checker = SemanticChecker(
        config_path="prompts/generator_config.yaml",
        function_definition={
            "functions": [
                {
                    "function": {
                        "name": "func1",
                        "description": "A dummy function for testing func1",
                    }
                },
                {
                    "function": {
                        "name": "func2",
                        "description": "A dummy function for testing func2",
                    }
                },
            ]
        },
    )
    return checker


@pytest.mark.asyncio
async def test_semantic_checker_run_valid(semantic_checker, mocker):
    mock_response_with_yes = mocker.AsyncMock(spec=ChatCompletions)
    mock_response_with_yes.choices = [
        mocker.AsyncMock(
            message=mocker.AsyncMock(content='{ "thought": "", "pass": "yes" }')
        )
    ]
    mocker.patch(
        "semantic_checker.async_inference", return_value=mock_response_with_yes
    )

    valid, invalid = await semantic_checker.run(
        [
            {
                "query": "test query",
                "function_calls": [{"name": "test_function", "args": {}}],
            }
        ]
    )

    assert len(valid) == 1
    assert len(invalid) == 0
    assert valid[0]["query"] == "test query"


@pytest.mark.asyncio
async def test_semantic_checker_run_invalid(semantic_checker, mocker):
    mock_response_with_no = mocker.AsyncMock(spec=ChatCompletions)
    mock_response_with_no.choices = [
        mocker.AsyncMock(
            message=mocker.AsyncMock(
                content='{ "thought": "Function call does not match query intent", "pass": "no" }'
            )
        )
    ]
    mocker.patch("semantic_checker.async_inference", return_value=mock_response_with_no)

    valid, invalid = await semantic_checker.run(
        [
            {
                "query": "test query",
                "function_calls": [{"name": "test_function", "args": {}}],
            }
        ]
    )

    assert len(valid) == 0
    assert len(invalid) == 1
    assert invalid[0]["query"] == "test query"
