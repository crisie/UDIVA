import argparse
from argparse import RawTextHelpFormatter
import sys
import os

sys.path.append("./lib")
from evaluation import evaluate_solution

def get_arguments():
    parser = argparse.ArgumentParser(description='-------- Dyadic Workshop: Behavior forecasting track - ICCV 2021 --------\n\nThis demo will evaluate your submission file.', formatter_class=RawTextHelpFormatter)
    parser.add_argument('-i', '--input', type=str, required=False, default="./input", help="Path to the input folder.")
    parser.add_argument('-o', '--output', type=str, required=False, default="./output", help="Folder where evaluation outputs will be stored.")
    parser.add_argument('-fp', '--face_percentage', type=float, default=0.1, help="Maximum normalized percentage for face.")
    parser.add_argument('-fb', '--face_bins', type=float, default=100, help="Bins considered for face.")
    parser.add_argument('-bp', '--body_percentage', type=float, default=0.5, help="Maximum normalized percentage for body.")
    parser.add_argument('-bb', '--body_bins', type=float, default=10, help="Bins considered for body.")
    parser.add_argument('-hp', '--hands_percentage', type=float, default=0.5, help="Maximum normalized percentage for hands.")
    parser.add_argument('-hb', '--hands_bins', type=float, default=100, help="Bins considered for hands.")
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    config = {
        "face_percentage": args.face_percentage,
        "face_bins": args.face_bins,
        "body_percentage": args.body_percentage,
        "body_bins": args.body_bins,
        "hand_percentage": args.hands_percentage,
        "hand_bins": args.hands_bins
    }
    evaluate_solution(args.input, args.output, config=config)

# python demo_evaluate.py -fp 0.2