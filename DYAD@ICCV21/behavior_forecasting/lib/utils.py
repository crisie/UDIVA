import sys
sys.path.append("../")
import numpy as np
import json
from annotations_parser import parse_face_at, parse_lhand_at, parse_rhand_at, parse_body_at
import os
from tqdm import tqdm
import pandas as pd
from config import *

def generate_baseline(annotations_folder, segments_path, input_folder):
    """
    :param annotations_folder: path to the annotations folder (with a subfolder per each session inside)
    :param input_folder: path to input folder with 'ref', 'res' and 'aux' folders

    This function generates a submission file with the "frozen" strategy.
    This strategy propagates the landmarks from the last visible frame to the 
    immediate next frames to predict.
    """

    res_folder = os.path.join(input_folder, "res")
    output_file = os.path.join(res_folder, PRED_FILENAME)

    assert os.path.exists(input_folder), f"Input folder '{input_folder}' does not exist."
    assert output_file.split(".")[-1] == "json", "Submission file must be a '.json' file."
    assert segments_path.split(".")[-1] == "csv", "Frames to predict should be stored in a '.csv' file."
    assert os.path.exists(segments_path), f"Frames to predict could not be read from '{segments_path}'."
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    print("[Baseline: frozen]")
    # we open the csv with the clips to be predicted
    predictions_df = pd.read_csv(segments_path)
    pbar = tqdm(total=len(predictions_df.index))

    submission_dict = {}
    for session, session_df in predictions_df.groupby("session"):
        session = f"{int(session):06d}"
        session_dict = {}
        for task, task_df in session_df.groupby("task"):
            task_dict = {}
            annotations_file = os.path.join(annotations_folder, session, task, ANNOTATIONS_FILE)
            for i in range(len(task_df.index)):
                frame_init = task_df.iloc[i]["init"]
                frame_final = task_df.iloc[i]["end"]

                # we extract the last set of landmarks available right before the gap to predict.
                # if the immediate previous frame has no landmarks, we predict -1.
                last_face = parse_face_at(annotations_file, frame_init - 1)[1]
                last_body = parse_body_at(annotations_file, frame_init - 1)[1]
                last_lhand = parse_lhand_at(annotations_file, frame_init - 1)[1]
                last_rhand = parse_rhand_at(annotations_file, frame_init - 1)[1]

                # we predict those landmarks for future frames
                for frame in range(frame_init, frame_final + 1):
                    task_dict[f"{frame:05d}"] = {
                        "face": last_face[:,:2].tolist() if last_face is not None else -1,
                        "body": last_body[:,:2].tolist() if last_body is not None else -1,
                        "left_hand": last_lhand[:,:2].tolist() if last_lhand is not None else -1,
                        "right_hand": last_rhand[:,:2].tolist() if last_rhand is not None else -1
                    }
                pbar.update()

            session_dict[task] = task_dict
        submission_dict[session] = session_dict

    with open(output_file, 'w') as json_file:
        json.dump(submission_dict, json_file, ensure_ascii=False)
    print(f"Submission file saved to '{output_file}'")

def generate_perfect_baseline(annotations_folder, segments_path, input_folder):
    """
    :param annotations_folder: path to the annotations folder (with a subfolder per each session inside)
    :param input_folder: path to input folder with 'ref', 'res' and 'aux' folders

    This function generates a submission file with the "perfect" strategy.
    This strategy simply stores the ground truth landmarks for the frames to predict.
    Therefore, the score expected for this baseline is 1.00
    """

    res_folder = os.path.join(input_folder, "res")
    output_file = os.path.join(res_folder, PRED_FILENAME)

    assert os.path.exists(input_folder), f"Input folder '{input_folder}' does not exist."
    assert output_file.split(".")[-1] == "json", "Submission file must be a '.json' file."
    assert segments_path.split(".")[-1] == "csv", "Frames to predict should be stored in a '.csv' file."
    assert os.path.exists(segments_path), f"Frames to predict could not be read from '{segments_path}'."
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    print("[Baseline: perfect]")
    # we open the csv with the clips to be predicted
    predictions_df = pd.read_csv(segments_path)
    pbar = tqdm(total=len(predictions_df.index))

    submission_dict = {}
    for session, session_df in predictions_df.groupby("session"):
        session = f"{int(session):06d}"
        session_dict = {}
        for task, task_df in session_df.groupby("task"):
            task_dict = {}
            annotations_file = os.path.join(annotations_folder, session, task, ANNOTATIONS_FILE)
            for i in range(len(task_df.index)):
                frame_init = task_df.iloc[i]["init"]
                frame_final = task_df.iloc[i]["end"]
                
                # we predict those landmarks for future frames
                for frame in range(frame_init, frame_final + 1):
                    last_face = parse_face_at(annotations_file, frame)[1]
                    last_body = parse_body_at(annotations_file, frame)[1]
                    last_lhand = parse_lhand_at(annotations_file, frame)[1]
                    last_rhand = parse_rhand_at(annotations_file, frame)[1]
                    task_dict[f"{frame:05d}"] = {
                        "face": last_face[:,:2].tolist() if last_face is not None else -1,
                        "body": last_body[:,:2].tolist() if last_body is not None else -1,
                        "left_hand": last_lhand[:,:2].tolist() if last_lhand is not None else -1,
                        "right_hand": last_rhand[:,:2].tolist() if last_rhand is not None else -1
                    }
                pbar.update()

            session_dict[task] = task_dict
        submission_dict[session] = session_dict

    with open(output_file, 'w') as json_file:
        json.dump(submission_dict, json_file, ensure_ascii=False)
    print(f"Submission file saved to '{output_file}'")
        
def generate_randomized_baseline(annotations_folder, segments_path, input_folder):
    """
    :param annotations_folder: path to the annotations folder (with a subfolder per each session inside)
    :param input_folder: path to input folder with 'ref', 'res' and 'aux' folders

    This function generates a submission file with the "frozen" strategy.
    This strategy propagates the landmarks from the last visible frame to the 
    immediate next frames to predict.
    """

    res_folder = os.path.join(input_folder, "res")
    output_file = os.path.join(res_folder, PRED_FILENAME)

    assert os.path.exists(input_folder), f"Input folder '{input_folder}' does not exist."
    assert output_file.split(".")[-1] == "json", "Submission file must be a '.json' file."
    assert segments_path.split(".")[-1] == "csv", "Frames to predict should be stored in a '.csv' file."
    assert os.path.exists(segments_path), f"Frames to predict could not be read from '{segments_path}'."
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    print("[Baseline: randomized frozen]")
    # we open the csv with the clips to be predicted
    predictions_df = pd.read_csv(segments_path)
    pbar = tqdm(total=len(predictions_df.index))

    submission_dict = {}
    for session, session_df in predictions_df.groupby("session"):
        session = f"{int(session):06d}"
        session_dict = {}
        for task, task_df in session_df.groupby("task"):
            task_dict = {}
            annotations_file = os.path.join(annotations_folder, session, task, ANNOTATIONS_FILE)
            for i in range(len(task_df.index)):
                frame_init = task_df.iloc[i]["init"]
                frame_final = task_df.iloc[i]["end"]

                # we extract the last set of landmarks available right before the gap to predict.
                # if the immediate previous frame has no landmarks, we predict -1.
                last_face = parse_face_at(annotations_file, frame_init - 1)[1]
                last_body = parse_body_at(annotations_file, frame_init - 1)[1]
                last_lhand = parse_lhand_at(annotations_file, frame_init - 1)[1]
                last_rhand = parse_rhand_at(annotations_file, frame_init - 1)[1]

                # we predict those landmarks for future frames
                for frame in range(frame_init, frame_final + 1):
                    frames_dist = frame - frame_init + 1
                    task_dict[f"{frame:05d}"] = {
                        "face": (last_face[:,:2] + np.random.randint(0, frames_dist, size=last_face[:,:2].shape)).tolist() if last_face is not None else -1,
                        "body": (last_body[:,:2] + np.random.randint(0, frames_dist, size=last_body[:,:2].shape)).tolist() if last_body is not None else -1,
                        "left_hand": (last_lhand[:,:2] + np.random.randint(0, frames_dist, size=last_lhand[:,:2].shape)).tolist() if last_lhand is not None else -1,
                        "right_hand": (last_rhand[:,:2] + np.random.randint(0, frames_dist, size=last_rhand[:,:2].shape)).tolist() if last_rhand is not None else -1
                    }
                pbar.update()

            session_dict[task] = task_dict
        submission_dict[session] = session_dict

    with open(output_file, 'w') as json_file:
        json.dump(submission_dict, json_file, ensure_ascii=False)
    print(f"Submission file saved to '{output_file}'")



def generate_submission_template(annotations_folder, segments_path, input_folder):
    """
    :param input_folder: path to input folder with 'ref', 'res' and 'aux' folders

    This function generates a submission file with the "frozen" strategy.
    This strategy propagates the landmarks from the last visible frame to the 
    immediate next frames to predict.
    """

    res_folder = os.path.join(input_folder, "res")
    output_file = os.path.join(res_folder, PRED_FILENAME)

    assert os.path.exists(input_folder), f"Input folder '{input_folder}' does not exist."
    assert output_file.split(".")[-1] == "json", "Submission file must be a '.json' file."
    assert segments_path.split(".")[-1] == "csv", "Frames to predict should be stored in a '.csv' file."
    assert os.path.exists(segments_path), f"Frames to predict could not be read from '{segments_path}'."
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    print("[Baseline: submission (random)]")
    # we open the csv with the clips to be predicted
    predictions_df = pd.read_csv(segments_path)
    pbar = tqdm(total=len(predictions_df.index))

    submission_dict = {}
    for session, session_df in predictions_df.groupby("session"):
        session = f"{int(session):06d}"
        session_dict = {}
        for task, task_df in session_df.groupby("task"):
            task_dict = {}
            annotations_file = os.path.join(annotations_folder, session, task, ANNOTATIONS_FILE)
            for i in range(len(task_df.index)):
                frame_init = task_df.iloc[i]["init"]
                frame_final = task_df.iloc[i]["end"]

                # we extract the last set of landmarks available right before the gap to predict.
                # if the immediate previous frame has no landmarks, we predict -1.
                last_face = parse_face_at(annotations_file, frame_init - 1)[1]
                last_body = parse_body_at(annotations_file, frame_init - 1)[1]
                last_lhand = parse_lhand_at(annotations_file, frame_init - 1)[1]
                last_rhand = parse_rhand_at(annotations_file, frame_init - 1)[1]

                # we predict those landmarks for future frames
                for frame in range(frame_init, frame_final + 1):
                    task_dict[f"{frame:05d}"] = {
                        "face": (np.random.randint(0, 720, size=last_face[:,:2].shape)).tolist() if last_face is not None else -1,
                        "body": (np.random.randint(0, 720, size=last_body[:,:2].shape)).tolist() if last_body is not None else -1,
                        "left_hand": (np.random.randint(0, 720, size=last_lhand[:,:2].shape)).tolist() if last_lhand is not None else -1,
                        "right_hand": (np.random.randint(0, 720, size=last_rhand[:,:2].shape)).tolist() if last_rhand is not None else -1
                    }
                pbar.update()

            session_dict[task] = task_dict
        submission_dict[session] = session_dict

    with open(output_file, 'w') as json_file:
        json.dump(submission_dict, json_file, ensure_ascii=False)
    print(f"Submission file saved to '{output_file}'")