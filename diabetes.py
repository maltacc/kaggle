#!/usr/bin/env python
# coding: utf-8

# Simple pipeline to preprocess data and train a model to predict diabetes in patients. Uses Azure Machine Learning.

# In[21]:


# Connect to workspace
import azureml.core
from azureml.core import Workspace 

ws = Workspace.from_config()
print(f"Using Azure ML {azureml.core.VERSION, ws.name}")


# In[23]:


# Prepare dataset of diabetes patients
from azureml.core import Dataset

default_ds = ws.get_default_datastore() 

if 'diabetes dataset' not in ws.datasets:
    default_ds.upload_files(
        files=['./data/diabetes.csv', './data/diabetes2.csv'],
        target_path = 'diabetes-data',
        overwrite=True,
        show_progress=True)

    # Create tabular dataset from datastore path
    tabular_dataset = Dataset.Tabular.from_delimited_files(path=(default_ds, 'diabetes-data/*.csv'))

    # Register dataset
    try: 
        tabular_dataset = tabular_dataset.register(workspace=ws, 
                                        name='diabetes dataset',
                                        description='diabetes data',
                                        tags = {'format': 'CSV'},
                                        create_new_version=True)
        print('Dataset registered.')
    except Exception as ex: 
        print(ex)
else: 
    print('Dataset already registered.')


# In[24]:


# Create scripts for pipeline
# Create folder for pipeline step files
import os 
experiment_folder = 'diabetes_pipeline'
os.makedirs(experiment_folder, exist_ok=True)
print(experiment_folder)


# In[25]:


get_ipython().run_cell_magic('writefile', '$experiment_folder/prep_diabetes.py', '\n# Script 1 reads from the diabetes dataset and preprocesses it\n#Import libraries\nimport os \nimport argparse\nimport pandas as pd \nfrom azureml.core import Run \nfrom sklearn.preprocessing import MinMaxScaler\n\n# Get params\nparser = argparse.ArgumentParser() \nparser.add_argument(\'--input-data\', type=str, dest=\'raw_dataset_id\', help=\'Raw dataset\')\nparser.add_argument(\'--prepped-data\', type=str, dest=\'prepped_data\', default=\'prepped_data\', help=\'Folder for results\')\nargs = parser.parse_args()\nsave_folder = args.prepped_data\n\n# Get experiment run context\nrun = Run.get_context()\n\n# Load the data that was passed as an input dataset\nprint("Loading...")\ndiabetes = run.input_datasets[\'raw_data\'].to_pandas_dataframe()\n\n# Log initial row count\nrow_count = len(diabetes)\nrun.log(\'raw_rows\', row_count)\n\n# Remove null values\ndiabetes = diabetes.dropna() \n\n# Normalize numeric columns\nscaler = MinMaxScaler()\nnum_cols = [\'Pregnancies\',\'PlasmaGlucose\',\'DiastolicBloodPressure\',\'TricepsThickness\',\'SerumInsulin\',\'BMI\',\'DiabetesPedigree\']\ndiabetes[num_cols] = scaler.fit_transform(diabetes[num_cols])\n\n# Log newly processed rows \nrow_count = len(diabetes)\nrun.log(\'processed_rows\', row_count)\n\n# Save data\nprint(\'Saving...\')\nos.makedirs(save_folder, exist_ok=True)\nsave_path = os.path.join(save_folder, \'data.csv\')\ndiabetes.to_csv(save_path, index=False, header=True)\n\n# End run\nrun.complete()\n')


# In[26]:


get_ipython().run_cell_magic('writefile', '$experiment_folder/train_diabetes.py', "\n# Script Step 2 trains the model\n# Import libraries\nfrom azureml.core import Run, Model \nimport argparse\nimport pandas as pd \nimport numpy as np\nimport os\nimport joblib\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.metrics import roc_auc_score, roc_curve \nimport matplotlib.pyplot as plt\n\n# Get params\nparser = argparse.ArgumenetParser()\n# --traning-folder references the folder where the prepped data was saved\nparser.add_argument('--training-folder', type=str, dest='training_folder', help='training data folder')\nargs = parser.parse_args()\ntraining_folder = args.training_folder \n\n# Get experiment run context\nrun = Run.get_context() \n\n# Load the prepped data file in the training folder\nfile_path = os.path.join(training_folder, 'data.csv')\ndiabetes = pd.read_csv(file_path)\n\n# Separate labels from features\nX, y = diabetes[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, diabetes['Diabetic'].values\n\n# Split data into training and validation sets \nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)\n\n# Train decision tree model\nmodel = DecisionTreeClassifier().fit(X_train, y_train)\n\n# Determine metrics\ny_hat = model.predict(X_test)\nacc = np.average(y_hat == y_test)\nprint('Accuracy: ', acc)\nrun.log('Accuracy', np.float(acc)) # accuracy\n\ny_scores = model.predict_proba(X_test)\nauc = roc_auc_score(y_test, y_scores[:,1])\nprint('AUC: ' + str(auc))\nrun.log('AUC', np.float(auc)) # AUC\n\nfpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])\nfig = plt.figure(figsize=(6,4))\nplt.plot([0,1], [0,1], 'k--')\nplt.plot(fpr, tpr)\nplt.xlabel('False Positive Rate')\nplt.ylabel('True Positive Rate')\nplt.title('ROC Curve')\nplt.log_image(name='ROC', plot=fig)\nplt.show() \n\n# Save trained model to outputs folder\nos.makedirs('outputs', exist_ok=True)\nmodel_file = os.path.join('outputs', 'diabetes_model.pkl')\njoblib.dump(value=model, filename=model_file)\n\n# Register model \nModel.register(workspace=run.experiment.workspace, \n                model_path = model_file, \n                model_name = 'diabetes_model',\n                tags={'Training context' : 'Pipeline'}, \n                properties={'AUC' : np.float(auc), 'Accuracy' : np.float(acc)})\n\n# End run\nrun.complete()\n")


# In[27]:


# Prepare compute environment for pipeline steps 

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException 

cluster_name = 'mguo1'

try: 
    pipeline_cluster = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing cluster')
except ComputeTargetException: 
    try: 
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_V2', max_nodes=2)
        pipeline_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
        pipeline_cluster.wait_for_completion(show_output=True)
    except Exception as ex: 
        print(ex)


# In[28]:


# Create run configuration 

from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import RunConfiguration

# Set up python env 
diabetes_env = Environment('diabetes-pipeline-env')
diabetes_env.python.user_managed_dependencies = False; 
diabetes_env.docker.enabled = True

# Create package dependencies
diabetes_packages = CondaDependencies.create(conda_packages=['scikit-learn','ipykernel','matplotlib','pandas','pip'],
                                             pip_packages=['azureml-defaults','azureml-dataprep[pandas]','pyarrow'])

diabetes_env.python.conda_dependencies = diabetes_packages

# Register python environment 
diabetes_env.register(workspace=ws)
registered_env = Environment.get(ws, 'diabetes-pipeline-env')

# New runconfig object created 
pipeline_run_config = RunConfiguration()
pipeline_run_config.target = pipeline_cluster

# Link environment to run configuration
pipeline_run_config.environment = registered_env

print('Run config created.')


# In[33]:


from azureml.pipeline.core import PipelineData
from azureml.pipeline.steps import PythonScriptStep

# Get training data
diabetes_ds = ws.datasets.get("diabetes dataset")

# Create a PipelineData for model folder
prepped_data_folder = PipelineData("prepped_data_folder", datastore=ws.get_default_datastore())

# 1) Run the data prep script
train_step = PythonScriptStep(name = "Prepare Data",
                                source_directory = experiment_folder,
                                script_name = "prep_diabetes.py",
                                arguments = ['--input-data', diabetes_ds.as_named_input('raw_data'),
                                             '--prepped-data', prepped_data_folder],
                                outputs=[prepped_data_folder],
                                compute_target = pipeline_cluster,
                                runconfig = pipeline_run_config,
                                allow_reuse = True)

# 2) run the training script
register_step = PythonScriptStep(name = "Train and Register Model",
                                source_directory = experiment_folder,
                                script_name = "train_diabetes.py",
                                arguments = ['--training-folder', prepped_data_folder],
                                inputs=[prepped_data_folder],
                                compute_target = pipeline_cluster,
                                runconfig = pipeline_run_config,
                                allow_reuse = True)

print("Pipeline steps defined")


# OK, you're ready build the pipeline from the steps you've defined and run it as an experiment.

# In[34]:


from azureml.core import Experiment
from azureml.pipeline.core import Pipeline
from azureml.widgets import RunDetails

# Construct pipeline
pipeline_steps = [train_step, register_step]
pipeline = Pipeline(workspace=ws, steps=pipeline_steps)
print("Pipeline is built.")

# Create experiment and run pipeline
experiment = Experiment(workspace=ws, name = 'diabetes-training-pipeline')
pipeline_run = experiment.submit(pipeline, regenerate_outputs=True)
print("Pipeline submitted for execution.")
RunDetails(pipeline_run).show()
pipeline_run.wait_for_completion(show_output=True)


# In[12]:


from azureml.core import Model

for model in Model.list(ws):
    print(model.name, 'version:', model.version)
    for tag_name in model.tags:
        tag = model.tags[tag_name]
        print ('\t',tag_name, ':', tag)
    for prop_name in model.properties:
        prop = model.properties[prop_name]
        print ('\t',prop_name, ':', prop)
    print('\n')


# In[13]:


# Publish pipeline from run
published_pipeline = pipeline_run.publish_pipeline(
    name="Diabetes_Training_Pipeline", description="Trains diabetes model", version="1.0")

published_pipeline


# In[14]:


rest_endpoint = published_pipeline.endpoint
print(rest_endpoint)


# In[15]:


# Call pipeline endpoint
from azureml.core.authentication import InteractiveLoginAuthentication

interactive_auth = InteractiveLoginAuthentication()
auth_header = interactive_auth.get_authentication_header()
print("Authentication header ready.")


# In[16]:


# Track pipeline 
import requests

experiment_name = 'Run_pipeline'

rest_endpoint = published_pipeline.endpoint
response = requests.post(rest_endpoint, 
                         headers=auth_header, 
                         json={"ExperimentName": experiment_name})
run_id = response.json()["Id"]
run_id


# In[17]:


from azureml.pipeline.core.run import PipelineRun

published_pipeline_run = PipelineRun(ws.experiments[experiment_name], run_id)
pipeline_run.wait_for_completion(show_output=True)


# In[18]:


# Retrain model periodically using newly generated data 

from azureml.pipeline.core import ScheduleRecurrence, Schedule

# Submit the Pipeline every Monday at 00:00 UTC
recurrence = ScheduleRecurrence(frequency="Week", interval=1, week_days=["Monday"], time_of_day="00:00")
weekly_schedule = Schedule.create(ws, name="weekly-diabetes-training", 
                                  description="Based on time",
                                  pipeline_id=published_pipeline.id, 
                                  experiment_name=experiment_name, 
                                  recurrence=recurrence)
print('Pipeline scheduled.')

