import json


def convert_string_to_bool(string):
    """Converts string to bool.

    Args:
        string: str, string to convert.

    Returns:
        Boolean conversion of string.
    """
    return False if string.lower() == "false" else True


def fix_arguments(arguments):
    """Fixes command line arguments dictionary in place.
    """
    # Fix tf_record_example_schema.
    arguments["tf_record_example_schema"] = json.loads(
        arguments["tf_record_example_schema"].replace(";", " ")
    )

    # Fix use_graph_mode.
    arguments["use_graph_mode"] = convert_string_to_bool(
        string=arguments["use_graph_mode"]
    )

    # Fix input_fn_autotune.
    arguments["input_fn_autotune"] = convert_string_to_bool(
        string=arguments["input_fn_autotune"]
    )

    # Fix preprocess_input.
    arguments["preprocess_input"] = convert_string_to_bool(
        string=arguments["preprocess_input"]
    )

    # Fix use_sample_covariance.
    arguments["use_sample_covariance"] = convert_string_to_bool(
        string=arguments["use_sample_covariance"]
    )
