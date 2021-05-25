# DYAD2021 - Behavior forecasting track

This directory contains a set of tools useful for the participants of the *Behavior Forecasting* track, which is part of the DYAD'21 challenge (ICCV 2021).

## Evaluation script

You will find a ```demo_evaluate.py``` which will help you evaluate your submission file against the ground truth file (*json* format). The ground truth file should be in the ```./INPUT/ref/ground_truth.json``` and the submission file should be in the ```./INPUT/res/predictions.json```. Please, make sure that the normalization distances *csv* file is inside the ```./INPUT/aux/distances.csv```. Then, you can run the script as follows:

````
usage: demo_evaluate.py [-h] [-i INPUT] [-o OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to the input folder.
  -o OUTPUT, --output OUTPUT
                        Folder where evaluation outputs will be stored.
````

The expected output is:

````
> RESULTS: (Face=0.081, Body=0.861. Hands=0.271)
Evaluation results successfully saved to 'output'.
````

You will find extra plots (```plots``` and ```meta``` folders, for the *png* plots and *csv* values, respectively) and detailed metrics (```scores.txt```) in the specified *output* folder. Also, bins and normalization percentages can be entered as script parameters, for which default values correspond to the ones used in the CodaLab evaluation script.


## Visualization script

In order to ease the task of visualizing the joints part of the evaluation, we provide the ```demo_visualize.py``` script, which generates the video with the landmarks drawn. You can run the script as follows:

````
usage: demo_visualize.py [-h] -a ANNOTATIONS -v VIDEOS [-o OUTPUT] -s SESSION -t TASK [-l]

optional arguments:
  -h, --help            show this help message and exit
  -a ANNOTATIONS, --annotations ANNOTATIONS
                        Path to the annotations folder.
  -v VIDEOS, --videos VIDEOS
                        Path to the dataset with the videos.
  -o OUTPUT, --output OUTPUT
                        Output folder where the generated video will be stored.
  -s SESSION, --session SESSION
                        Session to be visualized.
  -t TASK, --task TASK  Task to be visualized: {FC1_T, FC2_T} (for this track).
  -l, --only_landmarks  To generate a video with only landmarks (black background)
````

Please, note that the *annotations* and *videos* paths should correspond to a root folder containing one folder per session (e.g. *001080*, *001081*, etc). Additionally, you can specify the ```--only_landmarks``` to draw the landmarks over a black frame.