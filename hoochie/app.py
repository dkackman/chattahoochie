import logging
import re
import argparse


def get_type(module_name, type_name):
    module = __import__(module_name)
    return getattr(module, type_name)


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(rf'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


set_global_logging_level(logging.CRITICAL)


def _parse_args_from_argv() -> argparse.Namespace:
    """Parses arguments coming in from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-name",
        help="HuggingFace Transformers model name.",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cuda",
        help="The device to run the model on - cpu or cuda.",
    )
    parser.add_argument(
        "-l",
        "--load_in_8bit",
        default=False,
        type=bool,
        help="Load the model as 8 bit",
    )
    parser.add_argument(
        "-t",
        "--tokenizer_type",
        default="AutoTokenizer",
        help="The type name of the tokenizer to use",
    )
    parser.add_argument(
        "-mt",
        "--model_type",
        default="AutoModelForCausalLM",
        help="The type name of the model to use",
    )
    return parser.parse_args()
