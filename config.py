"""
Script to construct TensorFlow configuration protobuf files based on templates.
"""

import argparse
from string import Template


def main(settings):

    with open(settings.template_path, 'r') as file:
        raw_template = file.read()

    template = Template(raw_template)
    new_config = template.substitute(train_num_steps=settings.train_num_steps)

    with open(settings.output_path, 'w') as file:
        file.write(new_config)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--train_num_steps', type=int, default=200000)
    settings = parser.parse_args()
    main(settings)


if __name__ == "__main__":
    cli()
