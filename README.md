<<<<<<< HEAD
# RAMP starting kit on classifying EEG signals

## Description

This project is based on EEGs signals. Electroencephalography is a monitoring method to record electrical activity of the brain. Non-invasive electrodes are placed on different locations on the scalp and the signals are retrieved. 

The goal of this challenge is to be able to predict a seen digit (from 0 to 9) based on the signals recorded from the brain of a subject who saw the same digit, and so is thinking about it. Which leads to the name of our dataset "The MNIST of Brain Digits". 

It originally contains over a million brain signals, of 2 seconds each captured with the stimulus of seeing a digit form 0 to 9. These signals were captured using 4 different EEGs devices, NeuroSky Mindwave, emotiv EPOC, Interaxon Muse and Emotiv Insight, covering a total of 19 brain locations.

This is a multi classifications challenge, with a total of 11 classes: -1, 0, ... , 9.
## Preparing data

- The original "MindBigData" is split into 4 seperate files, with multiple lines describing one experience. To prepare the data in the desired format that we eventually worked with, it had to be aggregated, shuffled and split into testing and training set. The [prepare_data.py script](prepare_data.py) does that for you, if you have the original zipped folders and wish to try things out. If not, we already ran this and the resulting csv files will be found in the [data](/data) folder.

- The [problem.py script](problem.py) is then used to combine the data from each device, then again shuffle the row to get maximum randomness. It is responsible for loading the final data onto your notebook, as well as connecting to the Ramp server.

## Get started

- The [starting_kit](mind_data_starting_kit.ipynb) provides explanations, steps as well as our final [estimator](/submissions/starting_kit/estimator.py).

- To run test this estimator on your side, simply open a terminal in this folder and launch the command:

```bash
ramp-test --submissions starting_kit
```

You can then replace the [starting_kit folder](/submissions/starting_kit) with your own folder containing your estimator.

- Make sure to download all the packages necessary if you wish to run things locally. You can do so by simply running 

```bash
pip install -U -r requirements.txt
```

## Further details


If you have any problems with the Ramp command, you can use this for help:

```bash
ramp-test --help
```

- If you wish to know more about how the Ramp works, more detailes can be found [here](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)
=======
# DataCamp_EEG_DATA

Create a new folder named "mindbigdata-ep-v1.0" and put all the data files from the website in there then execute this code. 
>>>>>>> b6636564e44aedd281e8bcc88fce6db931f846aa
