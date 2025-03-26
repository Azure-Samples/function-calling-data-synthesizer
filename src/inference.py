import logging
import os
import random
from typing import Any, AsyncIterable

from azure.ai.inference.aio import ChatCompletionsClient
from azure.ai.inference.models import (
    ChatCompletions,
    StreamingChatCompletionsUpdate,
    UserMessage,
)
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from tenacity import (
    Retrying,
    before_sleep_log,
    stop_after_attempt,
    wait_random_exponential,
)

from custom_exceptions import AllClientsFailedError
from log_handlers import get_handlers

load_dotenv()


logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
for handler in get_handlers():
    logger.addHandler(handler)


async def async_inference(  # type: ignore
    messages: list[UserMessage],
    model_configuration: dict[str, Any],
) -> AsyncIterable[StreamingChatCompletionsUpdate] | ChatCompletions:
    """
    Performs asynchronous inference by sending messages to multiple clients with retry logic.
    Args:
        messages (list[type[ChatRequestMessage]]): The input messages to be sent for inference.
        model_configuration (dict[str, Any]): Configuration parameters for the model.
    Returns:
        response: The response from the first successful client.
    Raises:
        AllClientsFailedError: If all clients fail to respond after retries.
        Exception: If the inference call fails after several retries.
    """
    client_settings: list[dict[str, Any]]
    # NOTE: Add more endpoints if these still encounter token limit and if it's hard to increase the limit on each API Endpoint
    client_settings = [
        {
            "endpoint": os.environ.get("MODEL_API_BASE_URL1", ""),
            "credential": AzureKeyCredential(os.environ.get("MODEL_API_KEY1", "")),
        },
        {
            "endpoint": os.environ.get("MODEL_API_BASE_URL2", ""),
            "credential": AzureKeyCredential(os.environ.get("MODEL_API_KEY2", "")),
        },
    ]

    # Shuffle the client settings to achieve load balancing across endpoints.
    # This is useful when multiple users run the job against the same endpoints at the same time
    # This is done only once at initialization to ensure All endpoints have an equal chance to be called
    random.shuffle(client_settings)

    try:
        for attempt in Retrying(
            reraise=True,
            before_sleep=before_sleep_log(logger, logging.WARNING),
            wait=wait_random_exponential(multiplier=1, min=1, max=10),
            stop=stop_after_attempt(20),
        ):
            with attempt:
                for idx, client_setting in enumerate(client_settings):
                    endpoint_var: str = client_setting["endpoint"]
                    credential_val: AzureKeyCredential = client_setting["credential"]
                    try:
                        async with ChatCompletionsClient(
                            endpoint_var, credential_val
                        ) as client:
                            response = await client.complete(
                                messages=messages, **model_configuration
                            )
                            logger.debug(f"succeeded with API endpoint {idx}")
                            return response
                    except Exception as e:
                        error_type = type(e).__name__
                        error_message = str(e)
                        logger.warning(
                            f"Inference call failed on API endpoint {idx} with error type {error_type}"
                            f" and message: {error_message}."
                        )
                        logger.debug(
                            f"Failed response from API endpoint {idx}. whole messages: {messages}"
                        )
                else:
                    raise AllClientsFailedError(
                        f"All {client_settings} clients failed to respond in this attempt"
                    )
    except Exception:
        logger.exception(
            f"Inference call failed after several retries\nInput messages were: {messages}"
        )
        raise
