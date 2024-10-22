# domainadapt_segmentation
Repository for benchmarking code for domain adaptation of 3D segmentation models 

# installl 
 - use a new conda env with python 3.10 
 - git clone the repo and cd into it. 
 - run pip install -e .
# Dataset setup 
 - the code in ./domainadapt_segmentation/notebooks contains descriptions for how to process the datasets 
# Training models 
- Train models using 
- python3 -m domainadapt_segmentation.train --config_path ./domainadapt_segmentation/example_configs/train_baseline.json 
- You can specify your own configs in another directory 

# Testing Models 
- How to test the model is also as simple as specifying a config file.
- An example is found in  ./domainadapt_segmentation/example_configs/test_baseline.json. You must specify the following 
    - model_weight: path to model checkpoint trained using our code. 
    - output_dir: this is where the segmentations will be stored
    - metrics_paht: this is a directory where we will store the disce score
    - device: wich GPU to run inference on 
    - test set: the picklefile we will like to run inference on and evaluate performance 

# What are the pickle files 
-  We use monai dataloaders which expect the data to be given as a list of dictionarities 
- We have pkl_contents= (train_set,val_set,test_set)
- Each of the train test splits is a list of dictionaries 
- train_set = [
    {'image':PathToVolume.nii.gz,
    'label':PathToMaskVolume.nii.gz,
    'phase': 1 if contrast else 0},
    'PatientID':Useful For stat calculation latet
    ]