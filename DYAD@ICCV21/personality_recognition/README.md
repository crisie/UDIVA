The current directory contains:
- the evaluation script: "evaluate.py"
- an input folder containing necessary files for running the script: "input"
- an output folder that will contain the results after running the script: "output" 

The evaluation script can be run using the following command: \
python evaluate.py input output

Inside the “input” folder you can find the following two subfolders:
- "res": Here you have to put the predictions in CSV format and name them "predictions.csv". Inside this folder, you can find submission samples for both validation and test phases.
- "ref": In its original format this folder contains the ground truth values. You have to rename the file you need as "ground_truth.csv" before running the script. Note that the files that are currently there are not the real scores, but random values given to help you run the script. In order to evaluate your method, you might need to use part of the training set as validation and create the corresponding ground truth file in the format provided here. The validation and test ground truth will be released later, according to the challenge schedule.

The evaluation script will generate a “scores.txt” file and save it inside the “output” folder. It contains the score for each of the 5 (O, C, E, A, N) traits and the mean.

Output example: \
OPENMINDEDNESS_Z: 4.933202 \
CONSCIENTIOUSNESS_Z: 5.562610 \
EXTRAVERSION_Z: 4.671072 \
AGREEABLENESS_Z: 5.801077 \
NEGATIVEEMOTIONALITY_Z: 2.700661 \
MEAN: 4.733725
