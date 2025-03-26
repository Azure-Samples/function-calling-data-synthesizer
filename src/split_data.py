import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import jsonlines
from sklearn.model_selection import StratifiedShuffleSplit

from log_handlers import get_handlers

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
for handler in get_handlers():
    logger.addHandler(handler)


def _remove_single_data_points(
    class_names: list[str],
    high_level_class_names: list[str],
    synthesized_pairs: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """
    Remove data points that have only a single occurrence in the class_names list.
    Args:
        class_names (list[str]): A list of class names.
        high_level_class_names (list[str]): A list of high-level class names corresponding to the class_names.
        synthesized_pairs (list[dict[str, Any]]): A list of dictionaries containing synthesized data.
    Returns:
        tuple[list[str], list[str], list[dict[str, Any]]]:
            - Updated list of class names with single data points removed.
            - Updated list of high-level class names with single data points removed.
            - Updated list of synthesized data with single data points removed.
    """
    to_remove = []
    removed_items = []
    for class_name, count in Counter(class_names).items():
        if count < 2:
            to_remove.append(class_name)
    for class_name in to_remove:
        idx = class_names.index(class_name)
        removed_class = class_names.pop(idx)
        removed_high_level_class = high_level_class_names.pop(idx)
        removed_data = synthesized_pairs.pop(idx)
        removed_items.append(
            {
                "high_level_class": removed_high_level_class,
                "low_level_class": removed_class,
                "data": removed_data,
            }
        )
    logger.info(f"removed items due to only one data point: {removed_items}")
    return class_names, high_level_class_names, synthesized_pairs


def split_data(
    input_file_path: Path,
    train_output_path: Path,
    val_output_path: Path,
    test_size: float,
) -> None:
    """
    Splits data from an input file into training and validation sets, and writes them to specified output files.

    The function reads data from the input file, processes it to generate class names and high-level class names,
    removes single data points, and then splits the data into training and validation sets using stratified sampling.
    It logs the distribution of classes in the training and validation sets, calculates the ratio of training to
    validation samples for each class, and writes the resulting sets to the specified output files.
    Args:
        input_file_path (Path): Path to the input file containing the data to be split.
        train_output_path (Path): Path to the output file where the training set will be written.
        val_output_path (Path): Path to the output file where the validation set will be written.
        test_size (float): Proportion of the data to include in the validation set.
    Returns:
        None
    """
    stk = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
    with jsonlines.open(input_file_path) as reader:
        class_names = []
        high_level_class_names = []
        synthesized_pairs = []
        for line in reader:
            func_call = line["function_calls"][0]
            # to do more detailed split, use extra metadata
            class_name = (
                func_call["function_name"]
                + "_"
                + "_".join(sorted(func_call["arguments"].keys()))
            )
            high_level_class_names.append(func_call["function_name"])
            class_names.append(class_name)
            synthesized_pairs.append(line)

    class_names, high_level_class_names, synthesized_pairs = _remove_single_data_points(
        class_names, high_level_class_names, synthesized_pairs
    )

    train_set = []
    val_set = []
    train_class_counter: Counter[str] = Counter()
    val_class_counter: Counter[str] = Counter()
    train_high_level_class_counter: Counter[str] = Counter()
    val_high_level_class_counter: Counter[str] = Counter()
    for _, (train_index, val_index) in enumerate(
        stk.split(synthesized_pairs, class_names)
    ):
        for idx in train_index:
            train_class_counter[class_names[idx]] += 1
            train_high_level_class_counter[high_level_class_names[idx]] += 1
            train_set.append(synthesized_pairs[idx])
        for idx in val_index:
            val_class_counter[class_names[idx]] += 1
            val_high_level_class_counter[high_level_class_names[idx]] += 1
            val_set.append(synthesized_pairs[idx])

    logger.info(
        f"Train high level class distribution: {train_high_level_class_counter}"
    )
    logger.info(f"Train class distribution: {train_class_counter}")
    logger.info(f"Val high level class distribution: {val_high_level_class_counter}")
    logger.info(f"Val class distribution: {val_class_counter}")
    class_ratio = {
        cls: (
            train_class_counter[cls] / val_class_counter[cls]
            if val_class_counter[cls] != 0
            else float("inf")
        )
        for cls in train_class_counter
    }
    logger.info(f"train / val ratio {class_ratio}")
    with jsonlines.open(train_output_path, "w") as writer:
        writer.write_all(train_set)
    with jsonlines.open(val_output_path, "w") as writer:
        writer.write_all(val_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split data into training and validation sets."
    )
    parser.add_argument(
        "--input-file-path",
        type=Path,
        required=True,
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--train-output-path",
        type=Path,
        required=True,
        help="Path to the output training JSONL file.",
    )
    parser.add_argument(
        "--val-output-path",
        type=Path,
        required=True,
        help="Path to the output validation JSONL file.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Proportion of the dataset to include in the validation split.",
    )
    args = parser.parse_args()
    split_data(
        args.input_file_path,
        args.train_output_path,
        args.val_output_path,
        args.test_size,
    )
