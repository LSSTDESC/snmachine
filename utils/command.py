from argparse import ArgumentParser
from run_pipeline import Pipeline
import yaml


def process():
    parser = ArgumentParser(description="Define a list of params to run"
            "snmachine on a desired dataset. This can be provided with -d"
            "path/to/data")

    parser.add_argument('settings')
    parser.add_argument('--dataset', '-d', type=str)

    arguments = parser.parse_args()

    try:
        with open(arguments.settings) as f:
            params = yaml.load(f)
    except IOError:
        print("Invalid yaml file provided")
        exit()

    if arguments.dataset:
        myObject = Pipeline(**params, dataset=arguments.dataset)
        myObject.run_pipeline()
        # score = myObject.success_chance()
        # print(score)
    else:
        myObject = Pipeline(**params)
        myObject.run_pipeline()
        # score = myObject.success_chance()
        # print(score)


if __name__ == "__main__":
    process()
