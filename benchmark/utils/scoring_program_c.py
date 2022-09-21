import pandas as pd
import numpy as np
from itertools import cycle
import sys, os


def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    submit_dir = os.path.join(input_dir, 'res')
    truth_dir = os.path.join(input_dir, 'ref')

    if not os.path.isdir(submit_dir):
        print(f"Submit directory: '{submit_dir}' doesn't exist")
    elif not os.path.isdir(truth_dir):
        print(f"Truth directory: '{truth_dir}' doesn't exist")
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        submission_list = os.listdir(submit_dir)
        header = [i for i in range(7)]

        for submission in submission_list:
            if ".txt" in submission:
                predictions = pd.read_csv(os.path.join(submit_dir, submission),
                                          names=header, index_col=False)
                break
        else:
            raise FileNotFoundError("No .txt file found in submission")

        predictions = predictions.apply(np.argmax, axis=1).values

        # Load gt set
        split = 'test'
        targets = pd.read_csv(os.path.join(truth_dir, f'{split}_ground_truth.csv'),
                              names=['labels'], index_col=False).values.flatten()

        assert len(predictions) % len(targets) == 0

        nclasses = 6
        correct = [0 for _ in range(nclasses)]
        count = [0 for _ in range(nclasses)]

        for pred, targ in zip(predictions, cycle(targets)):
            if pred == targ:
                correct[targ - 1] += 1
            count[targ - 1] += 1

        accuracies = [cr / ct for cr, ct in zip(correct, count)]

        class_keys = ["AMCA_Pedestrian", "AMCA_Cyclist", "AMCA_Car", "AMCA_Truck",
                      "AMCA_Tram", "AMCA_Tricycle"]

        scores = {"AMCA": np.mean(accuracies)}
        for key, value in zip(class_keys, accuracies):
            scores[key] = value

        with open(os.path.join(output_dir, 'scores.txt'), 'w') as f:
            for key, value in scores.items():
                f.write(f"{key}: {value*100:.2f}\n")


if __name__ == '__main__':
    main()
