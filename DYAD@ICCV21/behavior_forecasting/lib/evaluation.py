import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm
import re
from config import *


def __check_landmarks_list(prediction):
    if type(prediction) == int and prediction == -1:
        return True
    elif type(prediction) != list: # no other integer value is accepted
        return False
    
    return np.sum([len(p) != 2 or 
                    (type(p[0]) != int and type(p[0]) != float) or 
                    (type(p[1]) != int and type(p[1]) != float)  for p in prediction]) == 0

def __check_submission_file(gt_json, pred_json):
    # We check the keys for sessions, tasks and frames:
    r_session = re.compile('^[0-9][0-9][0-9][0-9][0-9][0-9]$')
    r_task = re.compile('^FC[1-2]_T$')
    r_frame = re.compile('^[0-9][0-9][0-9][0-9][0-9]$')
    for session in pred_json.keys():
        if not r_session.match(session):
            raise Exception(f"\n[ERROR] Wrong key format '{session}' for a session.")
        for task in pred_json[session].keys():
            if not r_task.match(task):
                raise Exception(f"\n[ERROR] Wrong key format '{task}' for a task from session {session}.")
            for frame in pred_json[session][task].keys():
                if not r_frame.match(frame):
                    raise Exception(f"\n[ERROR] Wrong key format '{frame}' for a frame from session {session} and task {task}.")

                # we check all keys of body entities are there
                entities = pred_json[session][task][frame].keys()
                if len(entities) != len(BODY_ENTITIES) or not (np.array(sorted(entities)) == np.array(sorted(BODY_ENTITIES))).all():
                    raise Exception(f"\n[ERROR] Unexpected keys for frame {frame} from session {session} and task {task}.\n> Found: {list(entities)}\n> Expected: {BODY_ENTITIES}")

    all_gt_sessions = [f"{sess}_{task}" for sess in gt_json.keys() for task in gt_json[sess].keys()]
    all_pred_sessions = [f"{sess}_{task}" for sess in pred_json.keys() for task in pred_json[sess].keys()]
    intersection = list(set(all_gt_sessions) & set(all_pred_sessions))
    union = set(list(all_gt_sessions) + list(all_pred_sessions))

    # Checking missing or surplus of session/tasks.
    if len(intersection) < len(all_gt_sessions):
        missing = [miss for miss in all_gt_sessions if miss not in intersection]
        raise Exception(f"\n[ERROR] Missing {len(missing)} videos to predict: {missing}. Please, add them to your submission file and submit it again")
    if len(union) > len(all_gt_sessions):
        surplus = [err for err in union if err not in all_gt_sessions]
        raise Exception(f"\n[ERROR] {len(surplus)} surplus videos predicted: {surplus}. Please, remove them and repeat your submission.")

    # Checking missing or surplus of frames inside session/tasks.
    total_missing = []
    total_surplus = []
    for session in gt_json.keys():
        for task in gt_json[session].keys():
            frames_gt = gt_json[session][task].keys()
            frames_pred = pred_json[session][task].keys()
            intersection = list(set(frames_gt) & set(frames_pred))
            union = set(list(frames_gt) + list(frames_pred))
            if len(intersection) < len(frames_gt):
                missing = [miss for miss in frames_gt if miss not in intersection]
                total_missing.append((session, task, missing))
                #raise Exception(f"\n[ERROR] Missing {len(missing)} frames for ({session}, {task}): {missing}.")
            if len(union) > len(frames_gt):
                surplus = [err for err in union if err not in frames_gt]
                total_surplus.append((session, task, surplus))
                #raise Exception(f"\n[ERROR] {len(missing)} surplus frames predicted for ({session}, {task}): {surplus}.")

    errors = ""
    if len(total_missing) > 0:
        for (session, task, missing) in total_missing:
            errors += f"\n[ERROR] Missing {len(missing)} frames for ({session}, {task}): {missing}."
    if len(total_surplus) > 0:
        for (session, task, surplus) in total_surplus:
            errors += f"\n[ERROR] {len(surplus)} surplus frames predicted for ({session}, {task}): {surplus}."
    if errors != "":
        raise Exception(errors)

def __compute_metric(distances, percentage, num_bins, output_folder="data", name="plot"):
    plots_output_folder = os.path.join(output_folder, "plots")
    metadata_output_folder = os.path.join(output_folder, "meta")

    if not os.path.exists(plots_output_folder):
        os.makedirs(plots_output_folder)
    if not os.path.exists(metadata_output_folder):
        os.makedirs(metadata_output_folder)

    total = len(distances)
    distances = np.array(distances)
    thresholds = [percentage * i / num_bins for i in range(num_bins+1)]
    success_rates = [np.sum(distances <= th) / total for th in thresholds]

    fig = plt.figure(0)
    plt.plot(thresholds, success_rates)
    plt.ylim((0,1.05))
    plt.xlim((0, percentage))
    plt.title(name)
    plt.xlabel("Normalized distance")
    plt.ylabel("Correct frames proportion")
    plt.savefig(os.path.join(plots_output_folder, f"{name}.png"))
    fig.clf()

    csv_path = os.path.join(metadata_output_folder, f"{name}.csv")
    pd.DataFrame(data=list(zip(thresholds, success_rates)), columns=["thresholds", "value"]).to_csv(csv_path, index=False)
    return np.sum(success_rates) / len(success_rates)
    
def evaluate_solution(input_folder, output_folder, config=DEFAULT_CONFIG):
    """
    :param input_folder: folder with subfolders 'ref' (ground_truth.json), 'res' (predictions.json) and 'aux' (distances.csv)
    :param output_folder: folder where output files will be stored
    :param config: dictionary structure with information related to percentages and bins used for each metric

    This function evaluates a given solution against the annotations. Face, body and hands individual metrics are shown.
    Plots for CED, PCK and SR are stored in 'output_folder' for each part of the body.

    :return: 
    """
    # -------- Check for needed files --------
    # Distances for normalization - csv
    aux_path = os.path.join(input_folder, 'aux')
    distances_path = os.path.join(aux_path, DISTANCES_FILENAME)
    if not os.path.exists(distances_path):
        raise Exception(f"[ERROR] File '{DISTANCES_FILENAME}' not found at '{aux_path}'.")
    # Ground truth file
    ref_path = os.path.join(input_folder, 'ref')
    gt_path = os.path.join(ref_path, GT_FILENAME)
    if not os.path.exists(gt_path):
        raise Exception(f"[ERROR] File '{GT_FILENAME}' not found at '{gt_path}'.")
    # Predictions submission file
    res_path = os.path.join(input_folder, 'res')
    pred_path = os.path.join(res_path, PRED_FILENAME)
    if not os.path.exists(pred_path):
        raise Exception(f"[ERROR] File '{PRED_FILENAME}' not found at '{pred_path}'.")

    # We load the predictions and the submission json file
    print("Reading submission file...")
    with open(pred_path) as f:
        submission_json = json.load(f)
    print("Reading ground truth file...")
    with open(gt_path) as f:
        gt_json = json.load(f)
    d = pd.read_csv(distances_path)

    # We make sure the submission file is valid
    __check_submission_file(gt_json, submission_json)

    print("Starting evaluation...")
    face_dists, lhand_dists, rhand_dists, body_dists = [], [], [], [[] for i in range(BODY_JOINTS)]
    pbar = tqdm(total=len(gt_json.keys()))

    for session in gt_json.keys():
        for task in gt_json[session].keys():
            palm_width = float(d.loc[(d["session"] == int(session)) & (d["task"] == task), "palm"])
            interpupil_dist = float(d.loc[(d["session"] == int(session)) & (d["task"] == task), "interpupil"])
            head_size = float(d.loc[(d["session"] == int(session)) & (d["task"] == task), "head"])

            # we evaluate each frame
            for frame_key in gt_json[session][task].keys():
                if session not in submission_json or task not in submission_json[session] or frame_key not in submission_json[session][task]:
                    # no predictions for this frame => all infinite
                    face_dists.append(float('inf'))
                    lhand_dists.append(float('inf'))
                    rhand_dists.append(float('inf'))
                    for i in range(len(body_dists)):
                        body_dists.append(float('inf'))
                    continue

                frame_gt = gt_json[session][task][frame_key]
                frame_pred = submission_json[session][task][frame_key]

                # -------- FACE --------
                prediction, truth = frame_pred["face"], frame_gt["face"]
                to_eval = type(truth) != int
                predicted = "face" in frame_pred and type(prediction) != int
                if not __check_landmarks_list(prediction):
                    raise Exception(f"[ERROR] Wrong prediction for face: session {session}, task {task}, frame {frame_key}. Only lists of 2D coordinates or '-1' are accepted predictions.\nFound: '{prediction}'")

                if to_eval and predicted:
                    # there is prediction => evaluate it
                    if len(prediction) != len(truth):
                        raise Exception(f"[ERROR] Predictions and GT do not have the same number of landmarks for face: session {session}, task {task}, frame {frame_key}.")
                    dist = np.max(np.linalg.norm([np.array(p) - np.array(t) for (p, t) in zip(prediction, truth)], axis=1))
                    normalized_dist = dist / interpupil_dist
                    face_dists.append(normalized_dist)
                elif to_eval:
                    # no prediction => infinite
                    face_dists.append(float('inf'))

                # -------- LEFT HAND --------
                prediction, truth = frame_pred["left_hand"], frame_gt["left_hand"]
                to_eval = type(truth) != int
                predicted = "left_hand" in frame_pred and type(prediction) != int
                if not __check_landmarks_list(prediction):
                    raise Exception(f"[ERROR] Wrong prediction for left hand: session {session}, task {task}, frame {frame_key}. Only lists of 2D coordinates or '-1' are accepted predictions.\nFound: '{prediction}'")

                left_visible = type(truth) != int or truth != 0 # truth == 0 means not visible
                if to_eval and predicted:
                    # there is prediction => evaluate it
                    if len(prediction) != len(truth):
                        raise Exception(f"[ERROR] Predictions and GT do not have the same number of landmarks for left hand: session {session}, task {task}, frame {frame_key}.")
                    dist = np.max(np.linalg.norm([np.array(p) - np.array(t) for (p, t) in zip(prediction, truth)], axis=1))
                    normalized_dist = dist / palm_width
                    lhand_dists.append(normalized_dist)
                elif to_eval:
                    # no prediction => infinite
                    lhand_dists.append(float('inf'))

                # -------- RIGHT HAND --------
                prediction, truth = frame_pred["right_hand"], frame_gt["right_hand"]
                to_eval = type(truth) != int
                predicted = "right_hand" in frame_pred and type(prediction) != int
                if not __check_landmarks_list(prediction):
                    raise Exception(f"[ERROR] Wrong prediction for right hand: session {session}, task {task}, frame {frame_key}. Only lists of 2D coordinates or '-1' are accepted predictions.\nFound: '{prediction}'")

                right_visible = type(truth) != int or truth != 0 # truth == 0 means not visible
                if to_eval and predicted:
                    # there is prediction => evaluate it
                    if len(prediction) != len(truth):
                        raise Exception(f"[ERROR] Predictions and GT do not have the same number of landmarks for right hand: session {session}, task {task}, frame {frame_key}.")
                    dist = np.max(np.linalg.norm([np.array(p) - np.array(t) for (p, t) in zip(prediction, truth)], axis=1))
                    normalized_dist = dist / palm_width
                    rhand_dists.append(normalized_dist)
                elif to_eval:
                    # no prediction => infinite
                    rhand_dists.append(float('inf'))

                # -------- BODY --------
                prediction, truth = frame_pred["body"], frame_gt["body"]
                to_eval = type(truth) != int
                predicted = "body" in frame_pred and type(prediction) != int
                if not __check_landmarks_list(prediction):
                    raise Exception(f"[ERROR] Wrong prediction for body: session {session}, task {task}, frame {frame_key}. Only lists of 2D coordinates or '-1' are accepted predictions.\nFound: '{prediction}'")

                if to_eval and predicted:
                    # there is prediction => evaluate it
                    if len(prediction) != len(truth):
                        raise Exception(f"[ERROR] Predictions and GT do not have the same number of landmarks for body: session {session}, task {task}, frame {frame_key}.")
                    dist = np.linalg.norm([np.array(p) - np.array(t) for (p, t) in zip(prediction, truth)], axis=1)
                    normalized_dist = dist / head_size
                    for i in range(len(body_dists)):
                        if i not in [LEFT_WRIST_IDX, RIGHT_WRIST_IDX] or (i == LEFT_WRIST_IDX and left_visible) or (i == RIGHT_WRIST_IDX and right_visible):
                            # wrists only evaluated if hands are visible
                            body_dists[i].append(normalized_dist[i])
                elif to_eval:
                    # no prediction => infinite
                    for i in range(len(body_dists)):
                        if i not in [LEFT_WRIST_IDX, RIGHT_WRIST_IDX] or (i == LEFT_WRIST_IDX and left_visible) or (i == RIGHT_WRIST_IDX and right_visible):
                            # wrists only evaluated if hands are visible
                            body_dists[i].append(float('inf'))
        pbar.update()

    # we compute the final metric value.
    F = __compute_metric(face_dists, config["face_percentage"], config["face_bins"], output_folder=output_folder, name="face")
    B = 0
    body_parts = []
    for i in range(len(body_dists)):
        body_part = __compute_metric(body_dists[i], config["body_percentage"], config["body_bins"], output_folder=output_folder, name=BODY_NAMES[i])
        body_parts.append(body_part)
        B += body_part / len(body_dists)

    LH = __compute_metric(lhand_dists, config["hand_percentage"], config["hand_bins"], output_folder=output_folder, name="left_hand")
    RH = __compute_metric(rhand_dists, config["hand_percentage"], config["hand_bins"], output_folder=output_folder, name="right_hand")
    H = (LH + RH) / 2

    with open(os.path.join(output_folder, 'scores.txt'), 'w') as f:
        info = zip(["face", "body", "hands"], [F, B, H])
        for (name, score) in info:
            f.write(f'{name}: {score:.4f}\n')
        info = zip(BODY_NAMES + ["left_hand", "right_hand"], body_parts + [LH, RH])
        #f.write('\n')
        for (name, score) in info:
            f.write(f'{name}: {score:.4f}\n')

    print("\n")
    print(f"> RESULTS: (Face={F:.3f}, Body={B:.3f}. Hands={H:.3f})")
    print(f"Evaluation results successfully saved to '{output_folder}'.")