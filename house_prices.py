import math
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from tensorflow.python.data import Dataset
import visualizations as vi
from matplotlib import pyplot as plt



def split_train_data(dataframe):

    # shuffle data
    dataframe.reindex(np.random.permutation(dataframe.index))

    # To Do: get file name from the dataframe and create test files using that
    total_rows = dataframe.shape[0]
    train = dataframe.loc[:int(total_rows * 0.6), :]                                # 60% rows for training
    val = dataframe.loc[train.shape[0] : train.shape[0] + int(total_rows * 0.2), :] # 20% rows for validation
    test = dataframe.loc[train.shape[0] + val.shape[0]:, :]                         # 20% rows for testing
    return train, val, test

def get_input_features(dataframe):

    # select everything but the label / target (the value to be predicted)
    selected_features = dataframe[
                                    [ "longitude",
                                      "latitude",
                                      "housing_median_age",
                                      "total_rooms",
                                      "total_bedrooms",
                                      "population",
                                      "households",
                                      "median_income" ]]

    # synthesize new features

    # copy the dataframe to avoid modifying the input df
    proccessed_features = selected_features.copy()
    proccessed_features["rooms_per_person"] = proccessed_features["total_rooms"] / proccessed_features["population"]
    return proccessed_features

def get_targets(dataframe):
    # returns a dataframe for target values
    selected_targets = dataframe["median_house_value"]
    return pd.DataFrame(selected_targets.copy() / 1000, columns=["median_house_value"])

def input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of multiple features.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def train_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Trains a linear regression model of multiple features.

  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.

  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    training_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for validation.

  Returns:
    A `LinearRegressor` object trained on the training data.
  """

  periods = 10
  steps_per_period = steps / periods

  print("Steps per period: " + str(steps_per_period))

  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=construct_feature_columns(training_examples),
      optimizer=my_optimizer
  )

  # Input functions.
  training_input_fn = lambda: input_fn(training_examples, training_targets["median_house_value"], batch_size = batch_size)
  predict_training_input_fn = lambda: input_fn(training_examples, training_targets["median_house_value"], num_epochs=1, shuffle=False)
  predict_validation_input_fn = lambda: input_fn(validation_examples, validation_targets["median_house_value"], num_epochs=1, shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    print(" Period: " + str(period))
    print("     Training")
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )

    print("     Computing Predictions")
    # 2. Take a break and compute predictions.
    training_predictions = linear_regressor.predict(input_fn = predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])

    validation_predictions = linear_regressor.predict(input_fn = predict_validation_input_fn)
    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

    # Compute training and validation loss.
    print("     Computing Loss")
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))
    # Occasionally print the current loss.
    print("     RMSE: %0.2f" % (training_root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error)

  print("Model training finished.")
    # Output a graph of loss metrics over periods.
  vi.plot_rmse_side_by_side(training_rmse, validation_rmse)

  return linear_regressor


def construct_feature_columns(input_features):
      # since all our features are of numeric type, we can construct the TensorFlow Feature Columns
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("./data/california_housing_train.csv")

print("Get an overview of data to figure out some oddities and possible corruption")
# print(california_housing_dataframe.describe())

# create data sets for training, validation and testing
raw_train_data, raw_val_data, raw_test_data = split_train_data(california_housing_dataframe)

# Note: both of the following calls can be combined in a single method but is done this way to make it more clear
train_input_features = get_input_features(raw_train_data)
train_targets = get_targets(raw_train_data)

validation_input_features = get_input_features(raw_val_data)
validation_targets = get_targets(raw_val_data)


# visualizations should be almost same as we have randomized the data, if not then data partition is not uniform
vi.plot_coolwarm_side_by_side(train_input_features, train_targets, validation_input_features, validation_targets)

linear_regressor = train_model(0.00003, 500, 5, train_input_features, train_targets, validation_input_features, validation_targets)

test_input_features = get_input_features(raw_test_data)
test_targets = get_targets(raw_test_data)

predict_test_input_fn = lambda: input_fn(
                                         test_input_features,
                                         test_targets["median_house_value"],
                                         num_epochs=1,
                                         shuffle=False)

test_predictions = linear_regressor.predict(input_fn= predict_test_input_fn)
test_predictions = np.array([item['predictions'][0] for item in test_predictions])

root_mean_squared_error = math.sqrt(metrics.mean_squared_error(test_predictions, test_targets))
print("Final RMSE (on test data): %0.2f" % root_mean_squared_error)




