Instructions
============

# Generate sample data

Running `python example/example.py` allows to train a new action model or update an existing one. Each action model has 
its own folder in `examples`. The existing models include `move`, `perception`, `turn` and `pr2`. There are multiple 
general arguments to control the learning process:
* `-v, --verbose {debug,info,warning,error,critical}` sets the verbosity level for the current run
* `-e, --example <model-name>` learns/updates the action model in `examples/<model-name>`
* `-a, --args <arg> <value>` passes an action-specific argument to the respective `<model-name>` module. For accepted 
arguments, refer to the modules' documentations
* `--recent` will address the most recently generated folder (by a previous run). If not given, a new folder with 
a timestamp of the current run `examples/<example-name>/<run-id>` is created and the respective module's 
`generate_data(fp, args)` function is called to re-generate the data files, the result of which will be stored in 
`examples/example-name>/<run-id>/data/000-<model-name>.parquet`.
* `--learn` trains `JPT` models for the `<model-name>` action. The model object will be stored in 
`examples/example-name>/<run-id>/000-<example-name>.tree`
* `--modulelearn` allows to use a user-specified learning function instead of the default call of `learn_jpt` in 
`example/example.py`
* `--crossval` triggers the crossvalidation of multiple (predefined) model settings 
* `--plot` will trigger plotting of the action model `JPT` (twice, with and without variable plots). 
The result will be stored in `examples/example-name>/<run-id>/plots/000-<example-name>.svg` and 
`examples/example-name>/<run-id>/plots/000-<example-name>-nodist.svg`.
* `--min-samples-leaf <n> and --min-impurity-improvement <n>` passes the `min_samples_leaf` or 
`min_impurity_improvement` parameter to the learning function 
* `--obstacles` will add obstacle handling in all functions (where necessary)
* `--data` will trigger generating data/world plots by calling the `examples/<model-name>` module's `plot_data` 
function
* `--prune` triggers the `do_prune` function during the learning of the action model to influence the default behavior

To incorporate a new model in the system, add a new folder `<model-name>` in `examples/` and provide a 
`examples/<model-name>/<model-name>.py` module implementing the functions `init`, `generate_data`, `learn-jpt`, 
`plot_data`, and `tear_down`, each accepting two arguments: `fp`, a string representing the filepath to the current 
run of the `<model-namme>` and `args`, an argument object allowing to access the arguments passed to the call. 

# Run BayRoB web application

Calling

    $ ./run.sh

will pull the docker container for the `BayRoB` web application and start it. 
The application can be accessed in a web browser under [http://127.0.0.1:5005/bayrob/](http://127.0.0.1:5005/bayrob/).
The web application's documentation can be found in the `Documentation` tab of the app.