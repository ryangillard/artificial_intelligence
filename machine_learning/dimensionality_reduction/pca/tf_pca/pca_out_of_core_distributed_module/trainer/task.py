import json
import os

from trainer import cli_argument_reformat
from trainer import cli_parser
from trainer import model


if __name__ == "__main__":
    # Parse command line arguments.
    arguments = cli_parser.parse_command_line_arguments()

    # Unused args provided by service.
    arguments.pop("job_dir", None)
    arguments.pop("job-dir", None)

    # Fix formatting of command line arguments.
    cli_argument_reformat.fix_arguments(arguments)

    # Append trial_id to path if we are doing hptuning.
    # This code can be removed if you are not using hyperparameter tuning.
    arguments["output_dir"] = os.path.join(
        arguments["output_dir"],
        json.loads(
            os.environ.get(
                "TF_CONFIG", "{}"
            )
        ).get("task", {}).get("trial", ""))

    print(arguments)

    # Instantiate instance of model trainer.
    trainer = model.TrainModel(params=arguments)

    # Run the training job.
    trainer.train_model()
