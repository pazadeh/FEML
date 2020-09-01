# Surrogate modelling of protein networks using nanoFE simulations and machine learning

This repository contains the codes and trained models presented in paper titled "A NanoFE Simulation-based Surrogate Machine Learning Model to Predict Mechanical Functionality of Protein Networks from Live Confocal Imaging". The preprint of the paper can be found in: 

https://www.biorxiv.org/content/10.1101/2020.03.27.011239v2

In summary, this paper respresent a method for carrying out an automatic analysis of the structure-function relationship in cytoskeletal protein networks. This is done by combining computation confocal imaging, nanoFE simulations and unregulated data mapping (machine learning (ML)). The method includes 3 different steps:

  1. live 3D confocal imaging and image processing (for detailed explanation take a look at Asgharzadeh et al. 2018 Acta Biomaterialia), for extracting structural features of the biopolymers networks, 
  
  2. in-silico mechanical experiments for extracting the mechanical traits of the biopolymers networks, and
  3. gradient boosting models to map the structural features to the mechanical traits.

The ML models are used for an automatic classification and regression.

# Training the ML models
To train the ML models we used a 5-fold cross validation scheme. The input can be found in "input.xlsx". The models are trained using a grid search method and aiming for specific accuracy in both classification and regression task.

The training could be computationally costly. Therefore, running the trianing on a machine with multiple cores (here: 16) is suggested.
To train the classification model use the "Classification-tarin.py" code and to train the regression model use the "Regression-train.py"

The paper is at the moment under revision and this repo will be completed after publishing the paper.
