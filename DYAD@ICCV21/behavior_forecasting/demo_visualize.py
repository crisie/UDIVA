import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import imageio
import cv2
import os
from tqdm import tqdm

import sys
sys.path.append("./lib")
from config import *
from annotations_parser import is_valid, is_to_predict, parse_face_at, parse_lhand_at, parse_rhand_at, parse_body_at, parse_gaze_at, is_lhand_visible, is_rhand_visible
from visualization import *

def get_arguments():
    parser = argparse.ArgumentParser(description='-------- Dyadic Workshop: Behavior forecasting track - ICCV 2021 --------\n\nThis demo will help you to get familiar with the parsing of the annotations. The script generates a video with the visualization of the annotations.', formatter_class=RawTextHelpFormatter)
    parser.add_argument('-a', '--annotations', type=str, required=True, help="Path to the annotations folder.")
    parser.add_argument('-v', '--videos', type=str, required=True, help="Path to the dataset with the videos.")
    parser.add_argument('-o', '--output', type=str, required=False, default="./output", help="Output folder where the generated video will be stored.")
    parser.add_argument('-s', '--session', type=str, required=True, help="Session to be visualized.")
    parser.add_argument('-t', '--task', type=str, required=True, help="Task to be visualized.")
    parser.add_argument('-l', '--only_landmarks', action='store_true', default=False, help="To generate a video with only landmarks (black background)")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()
    
    annotations_path = os.path.join(args.annotations, args.session, args.task, ANNOTATIONS_FILE)
    video_path = os.path.join(args.videos, args.session, f"{args.task}.mp4")
    assert os.path.exists(annotations_path), f"There is no annotations file for the session and task specified: {(args.session, args.task)}."
    assert os.path.exists(video_path), f"There is no video file for the session and task specified: {(args.session, args.task)}"
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    print(f"Generating visualization for {args.session} - {args.task}")
    output_path = os.path.join(args.output, f"{args.session}_{args.task}{'' if not args.only_landmarks else '_onlylandmarks'}.mp4")
    reader = imageio.get_reader(video_path)
    writer = imageio.get_writer(output_path, fps=25)
    for i, frame in tqdm(enumerate(reader)):
        # We show a black frame instead if --only_landmarks set
        if args.only_landmarks:
            frame = np.zeros_like(frame)
        
        # If frame is valid, we print landmarks
        valid = is_valid(annotations_path, i)
        to_predict = is_to_predict(annotations_path, i)
        if valid and not to_predict:
            # We parse the face landmarks and show it. Note: 'valid' is only annotated for the validation and test sets
            landmarks, valid = parse_face_at(annotations_path, i)[1:]
            if valid:
                frame = draw_face(frame, landmarks, valid, size=3)

            # We parse the gaze vector and show it. For gaze visualization, ALL face landmarks need to be parsed and sent to 'draw_gaze_vector'.
            gaze = parse_gaze_at(annotations_path, i)
            landmarks = parse_face_at(annotations_path, i, all=True)[1]
            if gaze is not None:
                frame = draw_gaze_vector(frame, gaze, landmarks)

            # We parse the right hand landmarks and show them (only if hand is visible, i.e. not under the table or behind their body)
            # In the visualizer, use 'bbox="black"' to draw a black bbox around these landmarks so left/right hands can be distinguished
            landmarks, valid = parse_rhand_at(annotations_path, i)[1:]
            right_visible = is_rhand_visible(annotations_path, i)
            if right_visible and valid:
                frame = draw_hand(frame, landmarks, valid)#, bbox="black")

            # We parse the left hand landmarks and show them (only if hand is visible, i.e. not under the table or behind their body)
            # In the visualizer, use 'bbox="black"' to draw a black bbox around these landmarks so left/right hands can be distinguished
            landmarks, valid = parse_lhand_at(annotations_path, i)[1:]
            left_visible = is_lhand_visible(annotations_path, i)
            if left_visible and valid:
                frame = draw_hand(frame, landmarks, valid)#, bbox="white")

            # We parse the body joints and show them. Hands visibilities are used for the assessment of the wrists visibility.
            landmarks, valid = parse_body_at(annotations_path, i)[1:]
            if valid:
                frame = draw_body(frame, landmarks, valid, right_visible=right_visible, left_visible=left_visible)
        else: # not valid or to predict
            frame = np.zeros_like(frame)
                 
        # We draw a frame counter on top-left of the frames.
        cv2.putText(frame,f'{i:06d}', (60,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        writer.append_data(frame)

    writer.close()
    reader.close()

    print(f"Done! You can find your visualization in '{output_path}.'")
        