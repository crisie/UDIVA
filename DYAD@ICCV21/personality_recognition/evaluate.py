#!/usr/bin/env python

import errno
import os
import csv
import sys
import numpy as np


def read_csv_file(file_path):
    if not os.path.exists(file_path):
        raise Exception("File not found: " + file_path)

    file_dict = {}
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        row_idx = -1
        for row in csv_reader:
            row_idx += 1
            if len(row) != 6:
                raise Exception("Each row entry should have length 6: participant_id, O, C, E, A, N")
            if row_idx > 0:
                file_dict[row[0]] = [float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])]
    return file_dict


def compute_metrics(gt_dict, predictions_dict):
    gt_values, predicted_values = [], []
    for participant_id in gt_dict.keys():
        gt_values.append(gt_dict[participant_id])
        predicted_values.append(predictions_dict[participant_id])

    mse_per_class = np.mean(np.square(np.subtract(gt_values, predicted_values)), axis=0)
    return mse_per_class


def main():
    input_path = sys.argv[1]
    ref_path = os.path.join(input_path, 'ref')
    res_path = os.path.join(input_path, 'res')
    output_path = sys.argv[2]

    try:
        os.makedirs(output_path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    gt_file_path = os.path.join(ref_path, 'ground_truth.csv')
    gt_dict = read_csv_file(gt_file_path)

    predictions_file_path = os.path.join(res_path, "predictions.csv")
    predictions_dict = read_csv_file(predictions_file_path)

    if len(gt_dict) != len(predictions_dict):
        raise Exception("Different number of participants between ground truth (" + str(len(gt_dict)) + ") and predictions files (" + str(len(predictions_dict)) + ").")
    for participant_id in gt_dict.keys():
        if participant_id not in predictions_dict.keys():
            raise Exception("Participant ID " + str(participant_id) + " not found in " + predictions_file_path + ".")

    mse_per_class = compute_metrics(gt_dict=gt_dict, predictions_dict=predictions_dict)
    with open(os.path.join(output_path, 'scores.txt'), 'w') as f:
        for annotation_idx, annotation in enumerate(["OPENMINDEDNESS_Z", "CONSCIENTIOUSNESS_Z", "EXTRAVERSION_Z", "AGREEABLENESS_Z", "NEGATIVEEMOTIONALITY_Z"]):
            f.write('{}: {:f}\n'.format(annotation, mse_per_class[annotation_idx]))
        f.write('{}: {:f}\n'.format("MEAN", np.mean(mse_per_class)))


main()
