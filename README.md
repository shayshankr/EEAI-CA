### Engineering and Evaluating Artificial Intelligence (EEAI - CA)

## Task 1 – Chained Multi-Output Classification
- Three models are trained independently for Type2, Type3, and Type4 labels.
- Prediction happens in a chained manner:  
  Type2 → Type3 (if Type2 is correct) → Type4 (if both Type2 and Type3 are correct).
- Final accuracy is calculated based on how many predictions are fully correct across all three stages.
- Models: 'Random Forest', 'Logistic Regression'.


## Task 2 – Hierarchical Classification
- Prediction is done step-by-step using filtered data at each stage.
- Model 1 (Type2) predicts → filters data for Model 2 (Type3) → filters again for Model 3 (Type4).
- Each model is trained only on relevant subsets of data.
- Models: 'Random Forest', 'Logistic Regression'.
- Fully modular with runtime model selection per stage.


## How to Run
- Make sure you have Python 3.10+ installed.
- Install requirements
  - pip install -r requirements.txt
- Run the main controller
  - python main.py


## Dependencies
- Listed in 'requirements.txt'
- Main packages include:
  - scikit-learn
  - pandas
  - numpy
