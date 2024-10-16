"""
Step 1: Create a some PandasUDFs that generate synthetic datasets for each group.

Step 2: Configure a PandasUDF to train group-level models, log the models, register them
to UC, and perform some basic operations on the UC model version.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from pyspark.sql.types import StructType, StringType, FloatType
from pyspark.sql import SparkSession
import mlflow
from mlflow.client import MlflowClient

spark = SparkSession.builder.getOrCreate()

spark.conf.set('spark.sql.adaptive.enabled', 'false')
mlflow.autolog(disable=True)

groups = 3
n_features_per_group = 10
n_samples_per_group = 100

output_delta_location = 'main.default.mlc_pandasudf_models'
experiment_location  = '/mlc_pandasudf_experiment'

# Step 1
# Create some UDF that generate sythentic datasets for each group
def create_groups(groups=groups):

  groups = [[f'group_{str(n+1).zfill(2)}'] for n in range(groups)]

  schema = StructType()
  schema.add('group_name', StringType())

  return spark.createDataFrame(groups, schema=schema)



def get_feature_col_names(n_features_per_group=n_features_per_group):
  return [f"features_{n}" for n in range(n_features_per_group)]


def configure_features_udf(n_features_per_group=n_features_per_group, n_samples_per_group=n_samples_per_group):

  def create_group_features(group_data: pd.DataFrame) -> pd.DataFrame:

    features, target = make_regression(n_samples=n_samples_per_group, n_features=n_features_per_group)
    feature_names = get_feature_col_names()
    df = pd.DataFrame(features, columns=feature_names)

    df['target'] = target.tolist()

    group_name = group_data["group_name"].loc[0]
    df['group_name'] = group_name

    col_order = ['group_name'] + feature_names + ['target']

    return df[col_order]
  
  return create_group_features


spark_schema = StructType()
spark_schema.add('group_name', StringType())
for feature_name in get_feature_col_names():
  spark_schema.add(feature_name, FloatType())
spark_schema.add('target', FloatType())


# Step 2
# Use a PandasUDF to train a mdoel for each group, log it to an
# MLflow experiment, register the model as a UC model version, and
# do some basic operations on the UC model version
def fit_group_models(group_data: pd.DataFrame) -> pd.DataFrame:
  """
  Fit a regression model for the group and save it in pickle format to DBFS.
  """
  group_name = group_data["group_name"].loc[0] 

  mlflow.set_registry_uri("databricks-uc")
  mlflow.set_experiment(experiment_location)
  mlflow.autolog(log_models = True) 
  
  with mlflow.start_run(run_name = group_name) as run:
    group_features = group_data.drop(["group_name", "target"], axis=1)
    group_target = group_data['target']
    group_model = LinearRegression().fit(group_features.to_numpy(), np.array(group_target))

    model_name = f"main.default.mlc_regression_{group_name}"
    model_uri = f"runs:/{run.info.run_id}/model"
    model_registration = mlflow.register_model(model_uri, f"main.default.mlc_regression_{group_name}")
    model_version = model_registration.version

    client = MlflowClient()
    client.update_model_version(name=model_name,
                                version=model_version,
                                description="Adding a description")
    
    client.set_model_version_tag(model_name, model_version, "validation_status", "approved")

    df = pd.DataFrame([group_name], columns=['group_name'])

  return df

train_schema = StructType()
train_schema.add('group_name', StringType())


if __name__ == "__main__":

    # Generate the synthetic dataset
    groups = create_groups()
    udf = configure_features_udf()
    features = groups.groupBy('group_name').applyInPandas(udf, spark_schema)

    # Execute the model training and logger
    fitted_models = features.groupBy('group_name').applyInPandas(fit_group_models, schema=train_schema)
    fitted_models.write.mode('overwrite').format('delta').saveAsTable(output_delta_location)


