import pytest
from azure.ai.inference.models import (
    ChatCompletions,
    UserMessage,
)
from tenacity import (
    Retrying,
    stop_after_attempt,
    wait_random_exponential,
)

from custom_exceptions import AllClientsFailedError
from inference import async_inference


@pytest.fixture(autouse=True)
def mock_retrying(mocker):
    mocker.patch(
        "inference.Retrying",
        return_value=Retrying(
            reraise=True,
            wait=wait_random_exponential(multiplier=0.01, min=0.01, max=0.05),
            stop=stop_after_attempt(2),
        ),
    )


@pytest.mark.asyncio
async def test_async_inference_success(mocker):
    mock_client = mocker.patch("inference.ChatCompletionsClient")
    mock_response = mocker.AsyncMock(spec=ChatCompletions)
    mock_client.return_value.__aenter__.return_value.complete.return_value = (
        mock_response
    )

    messages = [UserMessage(content="test message")]
    model_configuration = {"temperature": 0.7}

    response = await async_inference(messages, model_configuration)

    assert response == mock_response
    mock_client.assert_called_once()


@pytest.mark.asyncio
async def test_async_inference_all_clients_fail(mocker):
    mock_client = mocker.patch("inference.ChatCompletionsClient")
    mock_client.return_value.__aenter__.return_value.complete.side_effect = Exception(
        "API error"
    )

    messages = [UserMessage(content="test message")]
    model_configuration = {"temperature": 0.7}

    with pytest.raises(AllClientsFailedError):
        await async_inference(messages, model_configuration)


@pytest.mark.asyncio
async def test_async_inference_retry_logic(mocker):
    mock_client = mocker.patch("inference.ChatCompletionsClient")
    mock_response = mocker.AsyncMock(spec=ChatCompletions)
    mock_client.return_value.__aenter__.return_value.complete.side_effect = [
        Exception("Temporary failure"),
        mock_response,
    ]

    messages = [UserMessage(content="retry test message")]
    model_configuration = {"temperature": 0.7}

    response = await async_inference(messages, model_configuration)

    assert response == mock_response
    assert mock_client.return_value.__aenter__.return_value.complete.call_count == 2
