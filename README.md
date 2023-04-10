# Shunt Prediction using LSTM for Subarachnoid Hemorrhage Patients
This project aims to predict the need for shunt in subarachnoid hemorrhage (SAH) patients from intensive care unit (ICU) using a Long Short-Term Memory (LSTM) model.

### Dataset
The dataset used for this project was obtained from the University Medical Center Hamburg-Eppendorf

### Model
We used an LSTM model to predict the need for shunt in SAH patients. The model takes multiple ICU parameters as input and outputs a binary classification (0 for no shunt needed, 1 for shunt needed). The LSTM model is trained on the training set and validated on the validation set to prevent overfitting in a nested-k-fold regime. The final model is evaluated on the test set to assess its performance.

### Requirements
1. Python 3
2. Scikit-learn
3. Pandas
4. NumPy
5. Pytorch


### Usage:
Clone this repository:

<pre>
git clone https://github.com/agschweingruber/sah.git
cd sah
</pre>

Install the required packages:
<pre>
cd ./training
pip install -r requirements.txt
</pre>


Run the **train_Shunt.ipynb** script to train the model

### Acknowledgments 
We thank the Department of Neurosurgery, Neurology, Neuroradiology and the Intensive Care Unit at the University Medical Center Hamburg-Eppendorf for their support.  
