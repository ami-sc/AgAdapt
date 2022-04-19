"""
XGBoost Model

This script trains XGBoost models using the Leave-One-Field-Out (LOFO) approach,
given a directory with master datasets and a master file with the model
specifications.

**** AgAdapt Project ****
"""


import os
import time
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from itertools import cycle
from threading import Thread
from colorama import Fore, Style


status = False
current = 0
total = 1


def tick_msg(text):
    """
    Prints a message after a green '✔'.

    Parameters
    ----------
    text : str
        Message to be printed.
    """
    print(Fore.GREEN + "[ ✔ ]" + Style.RESET_ALL + " " + text)


def info_msg(text):
    """
    Prints a message after a blue 'i'.

    Parameters
    ----------
    text : str
        Message to be printed.
    """
    print(Fore.CYAN + "[ i ]" + Style.RESET_ALL + " " + text)


def anim_loading(text):
    """
    Displays a loading animation.

    The animation will be displayed until the global variable "status" is set
    to false.

    Parameters
    ----------
    text : str
        Message to be printed along with the animation.
    """
    frames = ["●○○", "○●○", "○○●", "○●○"]

    for i in cycle(frames):
        if status:
            print(Fore.MAGENTA + "[" + i + "] " + Style.RESET_ALL + text,
                  end = "\r", flush = True)
            time.sleep(0.4)
        else:
            break


def anim_process(text):
    """
    Displays a loading animation and the current process being performed.

    The current process is given by the global variable "current". The total
    number of processes are given by the global variable "total". The animation
    will be displayed until the global variable "status" is set to false.

    Parameters
    ----------
    text : str
        Message to be printed along with the animation.
    """
    frames = ["●○○", "○●○", "○○●", "○●○"]

    for frame in cycle(frames):

        if status:
            print(Fore.MAGENTA + "[" + frame + "]" + Style.RESET_ALL + " "
                  + str(current) + "/" + str(total) + " "
                  + "(" + format((current / total) * 100, ".2f") + "%)"
                  + " | " + text, end = "\r", flush = True)

            time.sleep(0.4)

        else:
            print(Fore.MAGENTA + "[●●●]" + Style.RESET_ALL + " "
                  + str(total) + "/" + str(total) + " "
                  + "(" + format(100, ".2f") + "%)"
                  + " | " + text, end = "\n", flush = True)

            break


def cross_validation(parameters, d_train, random_seed):
    """
    Performs k-fold cross validation on an XGBoost Model. Validation will run
    for a maximum of 999 boosting rounds, or until no improvement is observed
    for at least 10 boosting rounds.

    Parameters
    ----------
    parameters : dict
        Parameter dictionary for the XGBoost Model validation.
    d_train : xgb.DMatrix
        DMatrix containing the training data to be used in the validation.
    random_seed : int
        Random seed to be used for reproducibility of the validation.

    Returns
    -------
    float
        The lowest Mean Absolute Error (MAE) obtained from the cross-validation.
    """
    # Run cross-validation test using current combination.
    cv_model = xgb.cv(
        parameters,
        d_train,
        num_boost_round = 999,
        seed = random_seed,
        nfold = 5,
        metrics = "mae",
        early_stopping_rounds = 10
    )

    # Retrieve the lowest MAE from the test.
    return cv_model["test-mae-mean"].min()


def train_model(lofo_field, predictor_params, phn_trait, master_df, test_size,
                random_seed, max_range):
    """
    Trains an XGBoost Model given a set of predictor features and a phenotype
    trait to be predicted. Model training will follow a Leave-One-Field-Out
    (LOFO) approach, where the data for a target field will be excluded from
    the training dataset and used exclusively for testing.

    Parameters
    ----------
    lofo_field : str
        Target field to be used exclusively for testing.
    predictor_params : list
        Features to be used as predictors in the model.
    phn_trait : str
        Phenotype trait to be predicted.
    master_df : pd.DataFrame
        Master Dataset containing both the predictor parameters and the target
        phenotype trait, on a per-field arrangement.
    test_size : float
        Test size to be used for the train_test_split() function.
    random_seed : int
        Random seed to be used for reproducibility of model training.
    max_range : int
        Maximum range to be used when tuning the max_depth and min_child_weight
        parameters of the model.

    Returns
    -------
    data_stats : list
        Calculated statistics for the training and testing datasets used in the
        model.
    performance_stats : list
        Calculated statistics about the performance of the model.
    cv_dfs : list
        List of DataFrames containing tested combinations for parameter tuning.
    tuned_params : list
        List containing the best values for each parameter of the model, as
        evaluated by parameter tuning.
    model_tuned : xgb.Booster
        Trained XGBoost Model.
    """
    global status, total, current

    # --------------------------------------------------------------------------
    # Data Selection
    # --------------------------------------------------------------------------

    model_df = master_df[["Field", phn_trait] + predictor_params]

    test_df = model_df.loc[model_df["Field"] == lofo_field].copy()
    model_df = model_df.loc[model_df["Field"] != lofo_field].copy()

    test_df.reset_index(drop = True, inplace = True)
    model_df.reset_index(drop = True, inplace = True)

    # --------------------------------------------------------------------------
    # Calculation of Data Statistics
    # --------------------------------------------------------------------------

    model_mean = model_df[phn_trait].mean()
    data_stats = [test_df[phn_trait].mean(), test_df[phn_trait].std(),
                  model_mean, model_df[phn_trait].std()]

    # --------------------------------------------------------------------------
    # Baseline Model
    # --------------------------------------------------------------------------

    performance_stats = []

    mean_array = np.ones(test_df.shape[0])
    mean_array = mean_array.dot(model_mean)

    performance_stats.append(mean_absolute_error(test_df[phn_trait],
                                                 mean_array))
    performance_stats.append(mean_squared_error(test_df[phn_trait],
                                                mean_array, squared = False))

    # --------------------------------------------------------------------------
    # DMatrix Generation
    # --------------------------------------------------------------------------

    d_test = xgb.DMatrix(test_df[predictor_params], label = test_df[phn_trait])

    x_train, x_tune, y_train, y_tune = train_test_split(
        model_df[predictor_params],
        model_df[phn_trait],
        test_size = test_size,
        random_state = random_seed
    )

    d_train = xgb.DMatrix(x_train, label = y_train)
    d_tune = xgb.DMatrix(x_tune, label = y_tune)

    # --------------------------------------------------------------------------
    # Default Parameter Dictionary
    # --------------------------------------------------------------------------

    parameters = {
        # Parameters for Tree Booster
        "max_depth": 6,
        "min_child_weight": 1,
        "eta": 0.3,
        "subsample": 1.0,
        "colsample_bytree": 1.0,

        # Learning Task Parameters
        "objective": "reg:squarederror",
        "eval_metric": "mae"
    }

    # --------------------------------------------------------------------------
    # max_depth & min_child_weight Parameter Tuning
    # --------------------------------------------------------------------------

    cv_dfs = []
    tuned_params = []

    md_values = range(1, max_range + 1)
    mcw_values = range(0, max_range + 1)

    best_mae = float("inf")
    best_md = None
    best_mcw = None

    cv_df = pd.DataFrame(np.nan, index = md_values, columns = mcw_values)

    current = 0
    total = len(md_values) * len(mcw_values)
    status = True
    loading = Thread(target = anim_process,
                     args = ("max_depth & min_child_weight Parameter Tuning",))
    loading.start()

    for md in md_values:
        for mcw in mcw_values:

            parameters["max_depth"] = md
            parameters["min_child_weight"] = mcw

            cv_mae = cross_validation(parameters, d_train, random_seed)
            cv_df.loc[md, mcw] = cv_mae

            if cv_mae < best_mae:
                best_mae = cv_mae
                best_md = md
                best_mcw = mcw

            current += 1

    status = False
    loading.join()
    tick_msg("Successfully tuned max_depth & min_child_weight parameters.")

    parameters["max_depth"] = best_md
    tuned_params.append(best_md)
    parameters["min_child_weight"] = best_mcw
    tuned_params.append(best_mcw)
    cv_dfs.append(cv_df)

    # --------------------------------------------------------------------------
    # subsample & colsample_bytree Parameter Tuning
    # --------------------------------------------------------------------------

    ssmpl_values = [i / 10 for i in range(1, 11)]
    cb_values = [i / 10 for i in range(1, 11)]

    best_mae = float("inf")
    best_ssmpl = None
    best_cb = None

    cv_df = pd.DataFrame(np.nan, index = ssmpl_values, columns = cb_values)

    current = 0
    total = len(ssmpl_values) * len(cb_values)
    status = True
    loading = Thread(target = anim_process,
                     args = ("subsample & colsample_bytree Parameter Tuning",))
    loading.start()

    for ssmpl in ssmpl_values:
        for cb in cb_values:

            parameters["subsample"] = ssmpl
            parameters["colsample_bytree"] = cb

            cv_mae = cross_validation(parameters, d_train, random_seed)
            cv_df.loc[ssmpl, cb] = cv_mae

            if cv_mae < best_mae:
                best_mae = cv_mae
                best_ssmpl = ssmpl
                best_cb = cb

            current += 1

    status = False
    loading.join()
    tick_msg("Successfully tuned subsample & colsample_bytree parameters.")

    parameters["subsample"] = best_ssmpl
    tuned_params.append(best_ssmpl)
    parameters["colsample_bytree"] = best_cb
    tuned_params.append(best_cb)
    cv_dfs.append(cv_df)

    # --------------------------------------------------------------------------
    # eta Parameter Tuning
    # --------------------------------------------------------------------------

    eta_values = [0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0.001]

    cv_df = pd.DataFrame(np.nan, index = ["MAE"], columns = eta_values)

    best_mae = float("inf")
    best_eta = None

    current = 0
    total = len(eta_values)
    status = True
    loading = Thread(target = anim_process,
                     args = ("eta Parameter Tuning",))
    loading.start()

    for eta in eta_values:

        parameters["eta"] = eta

        cv_mae = cross_validation(parameters, d_train, random_seed)
        cv_df.loc["MAE", eta] = cv_mae

        if cv_mae < best_mae:
            best_mae = cv_mae
            best_eta = eta

        current += 1

    status = False
    loading.join()
    tick_msg("Successfully tuned eta parameter.")

    parameters["eta"] = best_eta
    tuned_params.append(best_eta)
    cv_dfs.append(cv_df)

    # --------------------------------------------------------------------------
    # Model Training
    # --------------------------------------------------------------------------

    boosting_rounds = 999

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Finding best number of boosting rounds.",))
    loading.start()

    # Train a model using tuned parameters to find best number of rounds.
    tuned_model = xgb.train(
        parameters,
        d_train,
        num_boost_round = boosting_rounds,
        evals = [(d_tune, "Boosting_Rounds_Test")],
        early_stopping_rounds = 15,
        verbose_eval = False
    )

    status = False
    loading.join()
    tick_msg("Successfully found best number of boosting rounds.")

    # Update best number of rounds.
    boosting_rounds = tuned_model.best_iteration + 1
    tuned_params.append(boosting_rounds)

    evaluation_result = {}

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Training final model.",))
    loading.start()

    # Re-train model with appropriate number of rounds.
    model_tuned = xgb.train(
        parameters,
        d_train,
        num_boost_round = boosting_rounds,
        evals = [(d_tune, "Final_Model_MAE")],
        evals_result = evaluation_result,
        verbose_eval = False
    )

    train_mae = min(evaluation_result["Final_Model_MAE"]["mae"])
    performance_stats.append(train_mae)

    parameters["eval_metric"] = "rmse"

    evaluation_result = {}

    xgb.train(
        parameters,
        d_train,
        num_boost_round = boosting_rounds,
        evals = [(d_tune, "Final_Model_RMSE")],
        evals_result = evaluation_result,
        verbose_eval = False)

    train_rms = min(evaluation_result["Final_Model_RMSE"]["rmse"])
    performance_stats.append(train_rms)
    parameters["eval_metric"] = "mae"

    status = False
    loading.join()
    tick_msg("Successfully trained final model.")

    # --------------------------------------------------------------------------
    # Prediction and Testing
    # --------------------------------------------------------------------------

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Predicting in " + lofo_field + " field.",))
    loading.start()

    # Make a prediction using the tuned model and show the MAE and RMSE.
    prediction_mae = mean_absolute_error(model_tuned.predict(d_test),
                                         test_df[phn_trait])
    performance_stats.append(prediction_mae)
    prediction_rms = mean_squared_error(model_tuned.predict(d_test),
                                        test_df[phn_trait], squared = False)
    performance_stats.append(prediction_rms)

    status = False
    loading.join()
    tick_msg("Successfully predicted target phenotype trait.")

    return data_stats, performance_stats, cv_dfs, tuned_params, model_tuned


def model_summary(field, data_stats, performance_stats, cv_dfs, tuned_params,
                  save_path, model_code, model_name, units):
    """
    Generates a summary of the trained XGBoost Model in markdown format.

    Parameters
    ----------
    field : str
        LOFO field used in the model.
    data_stats : list
        Calculated statistics for the training and testing datasets used in the
        model.
    performance_stats : list
        Calculated statistics about the performance of the model.
    cv_dfs : list
        List of DataFrames containing tested combinations for parameter tuning.
    tuned_params : list
        List containing the best values for each parameter of the model, as
        evaluated by parameter tuning.
    save_path : str
        Path to .md file to save created summary.
    model_code : str
        Versioning code to be appended to the filename.
    model_name : str
        Given name of the XGBoost Model.
    units : str
        Units of the predicted phenotype trait.
    """
    summary_path = save_path + '/' + "README.md"
    summary = open(summary_path, "w")

    summary.write("# " + field + " - " + model_name + " Model Summary ["
                  + model_code + "]")
    summary.write("\n\n")

    summary.write("***")
    summary.write("\n\n")
    summary.write("### Model Performance")
    summary.write("\n\n")

    summary.write("- Baseline Model [MAE] = " +
                  "{:.4f}".format(performance_stats[0]) + '\n')
    summary.write("- Baseline Model [RMSE] = " +
                  "{:.4f}".format(performance_stats[1]) + '\n')
    summary.write("- Trained Model [MAE] = " +
                  "{:.4f}".format(performance_stats[2]) + '\n')
    summary.write("- Trained Model [RMSE] = " +
                  "{:.4f}".format(performance_stats[3]) + '\n')
    summary.write("- Prediction [MAE] = " +
                  "{:.4f}".format(performance_stats[4]) + '\n')
    summary.write("- Prediction [RMSE] = " +
                  "{:.4f}".format(performance_stats[5]) + '\n')

    summary.write("***")
    summary.write("\n\n")
    summary.write("### Dataset Statistics")
    summary.write("\n\n")

    summary.write("- LOFO Field [Mean] = " +
                  "{:.4f} ".format(data_stats[0]) + units + '\n')
    summary.write("- LOFO Field [Standard Deviation] = " +
                  "{:.4f} ".format(data_stats[1]) + units + '\n')
    summary.write("- Model Dataset [Mean] = " +
                  "{:.4f} ".format(data_stats[2]) + units + '\n')
    summary.write("- Model Dataset [Standard Deviation] = " +
                  "{:.4f} ".format(data_stats[3]) + units + '\n')

    summary.write("***")
    summary.write("\n\n")
    summary.write("### max_depth & min_child_weight Grid Search")
    summary.write("\n\n")

    cv_dfs[0].index.name = r"md \ mcw"
    summary.write(cv_dfs[0].to_markdown(tablefmt = "github"))
    summary.write("\n\n")

    summary.write("***")
    summary.write("\n\n")
    summary.write("### subsample & colsample_bytree Grid Search")
    summary.write("\n\n")

    cv_dfs[1].index.name = r"ssmpl \ cb"
    summary.write(cv_dfs[1].to_markdown(tablefmt = "github"))
    summary.write("\n\n")

    summary.write("***")
    summary.write("\n\n")
    summary.write("### eta Grid Search")
    summary.write("\n\n")

    cv_dfs[2].index.name = r"eta"
    summary.write(cv_dfs[2].to_markdown(tablefmt = "github"))
    summary.write("\n\n")

    summary.write("***")
    summary.write("\n\n")
    summary.write("### Tuned Parameters")
    summary.write("\n\n")

    summary.write("- max_depth = " + str(tuned_params[0]) + '\n')
    summary.write("- min_child_weight = " + str(tuned_params[1]) + '\n')
    summary.write("- subsample = " + str(tuned_params[2]) + '\n')
    summary.write("- colsample_bytree = " + str(tuned_params[3]) + '\n')
    summary.write("- eta = " + str(tuned_params[4]) + '\n')
    summary.write("- num_boost_round = " + str(tuned_params[5]) + '\n')

    summary.close()


def main():
    global status, current, total

    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "-mf", "--master_file",
        type = str,
        help = "Path to .txt file containing dataset specifications.",
        required = True)
    parser.add_argument(
        "-ts", "--test_size",
        type = float,
        help = "Percentage of given data to be used for cross validation.",
        required = True)
    parser.add_argument(
        "-s", "--seed",
        type = int,
        help = "Random seed to be used.",
        required = True)
    parser.add_argument(
        "-mx", "--max_range",
        type = int,
        help = "Maximum range (inclusive) for parameters used in hyperparameter"
               " tuning.",
        required = True)
    parser.add_argument(
        "-md", "--model_directory",
        type = str,
        help = "Path to directory that will store tuned models.",
        required = True)

    args = parser.parse_args()

    print("\n")

    field_list = []
    field_dfs = []
    phn_params = []
    model_names = []
    gen_predict = []
    tmp_predict = []
    model_code = ""

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Loading master datasets.",))
    loading.start()

    with open(args.master_file, "r") as master_file:

        for line in master_file:
            if line.startswith("SRC"):
                master_directory = line.split(",")[1].split()[0]
                fields = os.listdir(master_directory)

                for field in fields:
                    if field.endswith(".h5"):
                        field_list.append(field.split("_")[0])
                        field_path = master_directory + "/" + field
                        field_dfs.append(pd.read_hdf(field_path, "Master"))

            elif line.startswith("PHN"):
                phn_params.append(line.split(",")[1])
                model_names.append(line.split(",")[2].split("\n")[0])

            elif line.startswith("GEN"):
                gen_predict.append(line)

            elif line.startswith("TMP"):
                tmp_predict.append(line)

            elif line.startswith("MCD"):
                model_code = line.split(",")[1].split("\n")[0]

    master_df = pd.concat(field_dfs, ignore_index = True, sort = True)

    status = False
    loading.join()
    tick_msg("Successfully loaded master datasets.\n")

    gen_params = []

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Identifying target genotype parameters.",))
    loading.start()

    for predictor in gen_predict:
        predictor_specs = predictor.split(",")

        if predictor_specs[1] == "Latent Dimensions":

            num_features = predictor_specs[2].split("\n")[0]

            # Drop individuals with no latent dimension data.
            master_df.dropna(subset = ["LD_01"], inplace = True)

            # Get column names.
            lat_dim = master_df.loc[:, master_df.columns.str.startswith("LD")]
            lat_dim = lat_dim.columns.to_list()

            # Use maximum possible number of Latent Dimensions.
            if num_features == "MAX":
                gen_params = gen_params + lat_dim

            # Use target number of Latent Dimensions.
            else:
                gen_params = gen_params + lat_dim[0:int(num_features)]

        elif predictor_specs[1] == "Principal Components":

            num_features = predictor_specs[2].split("\n")[0]

            # Drop individuals with no Principal Components.
            master_df.dropna(subset = ["PC_001"], inplace = True)

            # Get column names.
            prin_comp = master_df.loc[:, master_df.columns.str.startswith("PC")]
            prin_comp = prin_comp.columns.to_list()

            # Use maximum possible number of Principal Components.
            if num_features == "MAX":
                gen_params = gen_params + prin_comp

            # Use target number of Principal Components.
            else:
                gen_params = gen_params + prin_comp[0:int(num_features)]

    status = False
    loading.join()
    tick_msg("Successfully identified target genotype parameters.\n")

    tmp_params = []

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Identifying target temperature features.",))
    loading.start()

    for predictor in tmp_predict:
        predictor_specs = predictor.split(",")

        if predictor_specs[1] == "Air Features":

            num_features = predictor_specs[2].split("\n")[0]

            # Drop individuals with no Air Temperature Features.
            master_df.dropna(subset = ["ATF_01"], inplace = True)

            # Get column names.
            atf = \
                master_df.loc[:, master_df.columns.str.startswith("ATF")].copy()
            # Drop features that are not shared by all individuals.
            atf.dropna(axis = "columns", inplace = True)
            atf = atf.columns.to_list()

            # Use maximum possible number of Air Temperature Features.
            if num_features == "MAX":
                tmp_params = tmp_params + atf

            # Use target number of Air Temperature Features.
            else:
                tmp_params = tmp_params + atf[0:int(num_features)]

        elif predictor_specs[1] == "Soil Features":

            num_features = predictor_specs[2].split("\n")[0]

            # Drop individuals with no Soil Temperature Features.
            master_df.dropna(subset = ["STF_01"], inplace = True)

            # Get column names.
            stf = \
                master_df.loc[:, master_df.columns.str.startswith("STF")].copy()
            # Drop features that are not shared by all individuals.
            stf.dropna(axis = "columns", inplace = True)
            stf = stf.columns.to_list()

            # Use maximum possible number of Soil Temperature Features.
            if num_features == "MAX":
                tmp_params = tmp_params + stf

            # Use target number of Soil Temperature Features.
            else:
                tmp_params = tmp_params + stf[0:int(num_features)]

    status = False
    loading.join()
    tick_msg("Successfully identified target temperature features.\n")

    # Drop individuals that do not have any phenotype data.
    master_df.dropna(subset = phn_params, inplace = True)

    # Recalculate field list.
    field_list = (master_df["Field"].unique())

    master_df.reset_index(drop = True, inplace = True)

    model_num = 0
    for phn_feature in phn_params:

        col_names = [
            "Baseline Model [MAE]",
            "Baseline Model [RMSE]",
            "Trained Model [MAE]",
            "Trained Model [RMSE]",
            "Prediction [MAE]",
            "Prediction [RMSE]"
        ]

        gen_stats = pd.DataFrame(np.nan, index = field_list,
                                 columns = col_names)
        env_stats = pd.DataFrame(np.nan, index = field_list,
                                 columns = col_names)
        gen_env_stats = pd.DataFrame(np.nan, index = field_list,
                                     columns = col_names)

        units = phn_feature.split(" ")[2].split("\n")[0]

        # Directory to store files for the current model.
        model_name = model_names[model_num]
        directory_path = args.model_directory + "/" + model_name
        os.mkdir(directory_path)

        for field in field_list:

            # Directory to store files for the current field.
            field_path = directory_path + "/" + field
            os.mkdir(field_path)

            # ------------------------------------------------------------------
            # Genome Model
            # ------------------------------------------------------------------

            info_msg(field + " " + model_name + " Genome Model Training")
            print("**************************************************")

            # Directory to store Genome Model.
            gen_model_path = field_path + "/G__Genome_Model"
            os.mkdir(gen_model_path)

            # Set parameters only to Genotype Data.
            predictor_params = gen_params

            # Train a model with given parameters.
            data_stats, performance_stats, cv_dfs, tuned_params, tuned_model = \
                train_model(field, predictor_params, phn_feature, master_df,
                            args.test_size, args.seed, args.max_range)

            status = True
            loading = Thread(target = anim_loading,
                             args = ("Saving model and generating summary.",))
            loading.start()

            # Save the trained model to .json file.
            model_path = gen_model_path + "/" + field + "_XGB_"\
                + model_name + "_Model__G.json"
            tuned_model.save_model(model_path)

            # Generate a summary of the hyperparameter tuning performance.
            model_summary(field, data_stats, performance_stats, cv_dfs,
                          tuned_params, gen_model_path, model_code,
                          model_name, units)

            gen_stats.loc[field] = performance_stats

            status = False
            loading.join()
            tick_msg("Successfully saved model and generated summary.")
            print("**************************************************\n\n")

            # ------------------------------------------------------------------
            # Environment Model
            # ------------------------------------------------------------------

            info_msg(field + " " + model_name + " Environment Model Training")
            print("**************************************************")

            # Directory to store Environment Model.
            env_model_path = field_path + "/E__Environment_Model"
            os.mkdir(env_model_path)

            # Set parameters only to Environmental Data.
            predictor_params = tmp_params

            # Train a model with given parameters.
            data_stats, performance_stats, cv_dfs, tuned_params, tuned_model = \
                train_model(field, predictor_params, phn_feature, master_df,
                            args.test_size, args.seed, args.max_range)

            status = True
            loading = Thread(target = anim_loading,
                             args = ("Saving model and generating summary.",))
            loading.start()

            # Save the trained model to .json file.
            model_path = env_model_path + "/" + field + "_XGB_" \
                + model_name + "_Model__E.json"
            tuned_model.save_model(model_path)

            # Generate a summary of the hyperparameter tuning performance.
            model_summary(field, data_stats, performance_stats, cv_dfs,
                          tuned_params, env_model_path, model_code,
                          model_name, units)

            env_stats.loc[field] = performance_stats

            status = False
            loading.join()
            tick_msg("Successfully saved model and generated summary.")
            print("**************************************************\n\n")

            # ------------------------------------------------------------------
            # Genome + Environment Model
            # ------------------------------------------------------------------

            info_msg(field + " " + model_name
                     + " Genome + Environment Model Training")
            print("**************************************************")

            # Directory to store Environment Model.
            gen_env_model_path = field_path + "/GE__Genome_Environment_Model"
            os.mkdir(gen_env_model_path)

            # Set parameters only to Environmental Data.
            predictor_params = gen_params + tmp_params

            # Train a model with given parameters.
            data_stats, performance_stats, cv_dfs, tuned_params, tuned_model = \
                train_model(field, predictor_params, phn_feature, master_df,
                            args.test_size, args.seed, args.max_range)

            status = True
            loading = Thread(target = anim_loading,
                             args = ("Saving model and generating summary.",))
            loading.start()

            # Save the trained model to .json file.
            model_path = gen_env_model_path + "/" + field + "_XGB_" \
                + model_name + "_Model__GE.json"
            tuned_model.save_model(model_path)

            # Generate a summary of the hyperparameter tuning performance.
            model_summary(field, data_stats, performance_stats, cv_dfs,
                          tuned_params, gen_env_model_path, model_code,
                          model_name, units)

            gen_env_stats.loc[field] = performance_stats

            status = False
            loading.join()
            tick_msg("Successfully saved model and generated summary.")
            print("**************************************************\n\n")

        gen_stats.to_csv(directory_path +
                         "/G__Genome_Model_Performance.csv")
        env_stats.to_csv(directory_path +
                         "/E__Environment_Model_Performance.csv")
        gen_env_stats.to_csv(directory_path +
                             "/GE__Genome_Environment_Model_Performance.csv")

        model_num += 1

    tick_msg("Done!")


if __name__ == "__main__":
    main()
