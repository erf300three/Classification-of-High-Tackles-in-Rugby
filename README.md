# Classification-of-High-Tackles-in-Rugby

**WARNING THIS PROJECT IS VERY LARGE AND UNZIPPED IT IS AROUND 75GB OF DATA**

This project aims to classify the legality of a tackle based on the domestic law variation. If there is anything in the 
code that is worth looking at it is the full pipeline results that can be found within the Model Evalation directory. 

We used GitHub Copilot throughout the project to help with documentation and as a productivity tool to  reduce the amount
of time spent writing repetitive lines of code. It was not used to write entire functions or sections of code due to the 
known unreliability of the code that it produces. Instead, it was often use to finish single lines of code of well known
structures such as a for loop. 

## Running the project 

Unfortunately, due to none of the data could be provided with the submissions for this project. This means there is no 
way to run the any of the code provided. This wouldnt be possible as some of the scripts still rely on absolute paths 
into my directory which was not able to be corrected due to time constraints. 

Instead we have provided most if not all of the results generated throughout the project found within other directories.

The requirements file and environmentation creation script does contain all of the requirements for the envrionment and 
will automatically create and launch the environment. 

## Directory Structure 

### Classification of High Tackles in Rugby 

This directory contains all of the source code for the project along with the required slurm batch compute jobs for the
project. All the important information is contained within the sripts subdirectory and is then split between the 
Python scripts and slurm scripts. 

### Datasets 

This directory is empty as we are unable to provide the datasets used to create the models at this time. 

### Kudu Output 

This directory contains all the slurm output scripts that were created for all of the slurm jobs that were run
throughout the whole project. These can be ignored unless you want to look through hundereds of millions of lines of logs. 

### Model evaluations 

This directory stores all of the evaluation results for most of the models trained through this project. It is split 
between the different model types: action recogniser, tackle_locator, pose classification. It also contains all of the 
full pipeline evalaution runs that were completed using the different Tackle Temporal Locator models. These contain 
all of the additional files as well as the results csv for each pipeline run. 

### Models 

This directory stores the models that were created during training. Some of the models created additional files during
training that might be worth a look. 

### Presentation Demo 

This contains the pipeline results for two tackle videos that were used as demo material for the presentation. 
