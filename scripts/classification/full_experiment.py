# An example of an experiment for using PyCaret for a classification problem.
import pandas as pd
from pycaret.classification import ClassificationExperiment

if __name__ == '__main__':
    # 1. Read data as Pandas Dataframe
    df = pd.read_csv("/home/amaldonado/Code/pycaret-tutorial/datasets/breast_cancer/wdbc.data", sep=',', header=None)
    columns = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst','perimeter_worst', 'area_worst', 'smoothness_worst','compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
    df.columns = columns
    print(df)

    # 2. Set up experiment. It initialize the training environment. It should be called before any other Pycaret function.
    # We'll be using the OOP API.
    # We should set at least data and target. The rest is optional.
    exp = ClassificationExperiment()
    exp.setup(data=df, target="diagnosis", session_id=1) # we'll use the default configuration

    # 2.5 If we want to get transformed data
    transformed_data = exp.get_config("X_train_transformed")
    print(transformed_data)
    
    # 3. Compare models and return the best one.
    print("Available models: ", exp.models())
    best_model = exp.compare_models()
    


