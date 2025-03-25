import json
from pathlib import Path

import pytest

from generate_synthetic_pairs import (
    QASampler,
    SyntheticDataGenerator,
    SyntheticDataGeneratorConfig,
    _extract_data,
)


@pytest.fixture
def qa_jsonl_file(tmp_path: Path) -> Path:
    data = [
        {
            "function_calls": [
                {
                    "function_name": "func1",
                    "arguments": {"action": "action1"},
                }
            ],
            "query": "query1",
        },
        {
            "function_calls": [
                {
                    "function_name": "func1",
                    "arguments": {"action": "action2"},
                }
            ],
            "query": "query2",
        },
        {
            "function_calls": [
                {"function_name": "func2", "arguments": {"app_name": "app_name1"}}
            ],
            "query": "query3",
        },
        {
            "function_calls": [
                {"function_name": "func2", "arguments": {"app_name": "app_name2"}}
            ],
            "query": "query4",
        },
    ]
    file_path = tmp_path / "test_function_calling_annotations.jsonl"
    with file_path.open("w") as f:
        for item in data:
            f.write(f"{json.dumps(item)}\n")
    return file_path


@pytest.fixture
def funcs_json_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "funcs.json"
    file_path.write_text(
        json.dumps(
            {
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
            }
        )
    )

    return file_path


class TestQASampler:
    def test_load(self, qa_jsonl_file: Path) -> None:
        sampler = QASampler(path=qa_jsonl_file)
        assert len(sampler.per_function_annotations) == 2
        assert len(sampler.per_function_annotations["func1"]) == 2
        assert len(sampler.per_function_annotations["func2"]) == 2

    def test_sample(self, qa_jsonl_file: Path) -> None:
        sampler = QASampler(path=qa_jsonl_file)
        samples = sampler.sample("func1", 1)
        assert len(samples) == 1
        assert samples[0]["function_calls"][0]["function_name"] == "func1"

    def test_sample_more_than_available(
        self, qa_jsonl_file: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        sampler = QASampler(path=qa_jsonl_file)
        sampler.sample("func1", 3)
        expected_message = (
            "Wanted to sample 3 cases but only 2 annotations exists for func1. "
            "Sampling all annotations (2 cases) for func1"
        )
        assert any(expected_message in record.message for record in caplog.records)


class TestExtractData:
    def test_extract_valid_data(self) -> None:
        message_content = """{"query": "query1", "function_calls": [{"function_name": "func1", "arguments": {"action": "action1"}}]}
        {"query": "query2", "function_calls": [{"function_name": "func2", "arguments": {"app_name": "app_name1"}}]}"""
        cleaned, error_counts = _extract_data(message_content)
        assert len(cleaned) == 2
        assert cleaned[0]["query"] == "query1"
        assert cleaned[1]["query"] == "query2"
        assert error_counts.no_json_section == 0
        assert error_counts.json_decode_error == 0
        assert error_counts.missing_keys == 0

    def test_extract_data_with_no_json_section(self) -> None:
        message_content = """This is not a JSON line
        {"query": "query1", "function_calls": [{"function_name": "func1", "arguments": {"action": "action1"}}]}"""
        cleaned, error_counts = _extract_data(message_content)
        assert len(cleaned) == 1
        assert cleaned[0]["query"] == "query1"
        assert error_counts.no_json_section == 1
        assert error_counts.json_decode_error == 0
        assert error_counts.missing_keys == 0

    def test_extract_data_with_json_decode_error(self) -> None:
        message_content = """{"query": "query1", "function_calls": [{"function_name": "func1", "arguments": {"action": "action1"}}]}
        {"query": "query2", "function_calls": [{"function_name": "func2", "arguments": {"app_name": "app_name1"}}]"""
        cleaned, error_counts = _extract_data(message_content)
        assert len(cleaned) == 1
        assert cleaned[0]["query"] == "query1"
        assert error_counts.no_json_section == 0
        assert error_counts.json_decode_error == 1
        assert error_counts.missing_keys == 0

    def test_extract_data_with_missing_keys(self) -> None:
        message_content = """{"query": "query1", "function_calls": [{"function_name": "func1", "arguments": {"action": "action1"}}]}
        {"query": "query2"}"""
        cleaned, error_counts = _extract_data(message_content)
        assert len(cleaned) == 1
        assert cleaned[0]["query"] == "query1"
        assert error_counts.no_json_section == 0
        assert error_counts.json_decode_error == 0
        assert error_counts.missing_keys == 1


def test_prepare_user_message(mocker, qa_jsonl_file: Path, funcs_json_file: Path):
    config_path = Path("prompts/generator_config.yaml")
    real_config = SyntheticDataGeneratorConfig.from_yaml(config_path)
    mocker.patch(
        "generate_synthetic_pairs.QASampler.sample",
        return_value=[{"query": "dummy_query", "function_calls": []}],
    )
    generator = SyntheticDataGenerator(
        config=real_config,
        qa_jsonl_path=qa_jsonl_file,
        function_definitions_path=funcs_json_file,
    )
    result = generator._prepare_user_message("myFunc", {"desc": "some"}, 2)
    assert "`myFunc`" in result
    assert '"some"' in result
    assert " 2 " in result
    assert '"dummy_query"' in result
    assert "{{persona_instruction}}" not in result


def test_prepare_user_message_with_personality(
    mocker, qa_jsonl_file: Path, funcs_json_file: Path
):
    config_path = Path("prompts/generator_config.yaml")
    real_config = SyntheticDataGeneratorConfig.from_yaml(config_path)
    real_config.personality["enabled"] = True

    # Force the random choice to return a known personality
    mocker.patch("random.choice", return_value="friendly")

    # Enable personality
    generator = SyntheticDataGenerator(
        config=real_config,
        qa_jsonl_path=qa_jsonl_file,
        function_definitions_path=funcs_json_file,
    )

    result = generator._prepare_user_message("func1", {"desc": "some"}, 2)
    # Assert that the chosen persona appears in the generated message
    assert " friendly " in result
    assert "{{persona_instruction}}" not in result
