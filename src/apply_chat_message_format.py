import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from log_handlers import get_handlers

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
for handler in get_handlers():
    logger.addHandler(handler)


def apply_chat_message_format(input_path: Path, output_path: Path) -> None:
    """perform the conversion to chat message format used by fine-tuning

    Below is a sample output after conversion (one-line in JSONL file):

    ````json
    {
        "messages": [
            { "role": "system", "content": "You are an in-car assistant with a list of tools. ..." },
            { "role": "user", "content": "Play the song 'Starlight' on MusicBox" },
            { "role": "assistant", "content": "[{\"function_name\": \"play_audio_track\", \"arguments\": {\"service\": \"MusicBox\", \"media_type\": \"track\", \"title\": \"Starlight\"}}]" }
        ]
    }

    Args:
        input_path (Path): path to the input file
        output_path (Path): path to the output file
    """
    with open("prompts/system_prompt.txt", "r") as prompt_file:
        system_prompt = prompt_file.read()
    with input_path.open("r") as f:
        for line in f:
            json_load = json.loads(line)
            query = json_load["query"]
            if json_load["function_calls"] == []:
                logger.warning(f"Function call not found for {query}")
                continue
            else:
                function = json_load["function_calls"]

            with output_path.open("a") as output:
                system_dict = {"role": "system", "content": system_prompt}
                user_dict = {"role": "user", "content": query}
                assistant_content = json.dumps(function)
                assistant_dict = {"role": "assistant", "content": assistant_content}
                phi_fine_tuning_format = {
                    "messages": [system_dict, user_dict, assistant_dict]
                }

                output.write(json.dumps(phi_fine_tuning_format) + "\n")


def parse_arguments() -> Namespace:
    """
    Parse command-line arguments.

    Returns:
        Namespace: Parsed arguments.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=Path,
        help="The input file to convert to phi format.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default="fine_tuning_format_conversion.jsonl",
        help="The output file to save the fine-tuning format data to.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    apply_chat_message_format(input_path=args.input_path, output_path=args.output_path)
