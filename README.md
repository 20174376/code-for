# code-for-paper

This repository is still being updated.

# Unconstrained Positive Reframing

## Training model

run-t5.py for training T5, run-bart.py for training BART. Choose "unconstrained".

## Generate files

Use predict-used.py for it. And the default parameter settings are given. Choose "unconstrained".

## Evaluate files

Just use evaluate.py for it. Change parameters as required.

# Controlled Positive Reframing

run-t5.py for training T5, run-bart.py for training BART. Choose "controlled".

## Generate files

Use predict-used.py for it. And the default parameter settings are given. Choose "controlled".

## Re-ranking

After getting the file, you can use Classfication\score.py to score the candidate sentence, but change the "file_path".

## Evaluate files

Before the evaluation, for the sampling strategy (Top-k, Top-p and Typical-p), it is necessary to combine the files first. You can refer to combinefile.py. Then you can use evaluate.py for it. Change parameters as required. Remember change "--manyflag" to True and "--numbercount" to number of candidate sentences you returned for each original sentence, if you want to re-rank.

# RTQE

Enter the JudgeReframe folder, then you can train the model through main.py, but change the "model_path" and "save_path".

# Reframe Strategy Classfication

Enter the Classfication folder, then you can train the model through main.py, but change the "model_path" and "save_path".







