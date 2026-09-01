# Reddit Graduate Admissions Distress Study

Code for preparing and evaluating the Reddit graduate-admissions study data.

Research data, human labels, internal reviewer keys, model predictions, and
generated outputs are intentionally excluded from Git. Transfer those files to
approved server storage separately and pass their locations to scripts through
command-line arguments or configuration.

The active code is organized under `Code/`. Each stage is kept modular so that
cleaning, anchor classification, exposure construction, covariate construction,
and outcome analysis can run independently.
