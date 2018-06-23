# House Price Prediction
A project to get familiar with multiple layers of TensorFlow

- TensorFlow Estimator APIs
- TensorFlow custom layers

## Instructions for setting up virutalenv
cd ~/house_price_prediction

### Install python3
```
brew install python3
pip3 install virtualenv
```

### setup the virutalenv in the project's root directory
This tells virtualenv to create a an environment in .env/python3 directory and -p flag specifies which version of python to use.

`virtualenv -p  $(readlink `which python3`) .env/python3`

### Activate the virtual environment.
This is needed anytime we navigate back and forth between project directories

`source .env/python3/bin/activate`

### Install the dependencies
`pip install -r ./requirements.txt`
