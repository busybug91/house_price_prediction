import numpy as np
import pandas as pd
import tensorflow as tf
import visualizations as vi


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

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
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


def construct_feature_columns(input_features):
      # since all our features are of numeric type, we can construct the TensorFlow Feature Columns
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("./data/california_housing_train.csv")

print("Get an overview of data to figure out some oddities and possible corruption")
print(california_housing_dataframe.describe())

# create data sets for training, validation and testing
raw_train_data, raw_val_data, raw_test_data = split_train_data(california_housing_dataframe)

# Note: both of the following calls can be combined in a single method but is done this way to make it more clear
train_input_features = get_input_features(raw_train_data)
train_targets = get_targets(raw_train_data)

validation_input_features = get_input_features(raw_val_data)
validation_targets = get_targets(raw_val_data)


# visualizations should be almost same as we have randomized the data, if not then data partition is not uniform
vi.plot_coolwarm_side_by_side(train_input_features, train_targets, validation_input_features, validation_targets)




