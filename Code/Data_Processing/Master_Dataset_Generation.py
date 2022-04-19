"""
Master Dataset Generation

This script generates concatenated, per-field datasets containing a target
combination of Genotype, Phenotype, and Environmental variables.

**** AgAdapt Project ****
"""


import os
import time
import argparse
import pandas as pd
import numpy as np
from functools import reduce
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


def phenotype_data(phn_df, phn_var):
    """
    Retrieves specified traits from the Phenotype Data of a target field.

    Parameters
    ----------
    phn_df : pd.DataFrame
        DataFrame containing target Phenotype Data.
    phn_var : pd.DataFrame
        Phenotype traits to be extracted from the given DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame containing target phenotype traits.
    """
    # Get target Phenotype Data for specified field.
    field_phn = phn_df[phn_var].copy()
    field_phn.rename(columns = {"Field-Location": "Field"}, inplace = True)
    field_phn.sort_values(by = ["Pedigree"], inplace = True)
    field_phn.reset_index(drop = True, inplace = True)

    return field_phn


def genotype_data(gen_df, field_phn):
    """
    Retrieves Genotype Data for each individual in a Phenotype DataFrame.

    Parameters
    ----------
    gen_df : pd.DataFrame
        DataFrame containing target Genotype Data.
    field_phn : pd.DataFrame
        DataFrame containing a list of individuals, along with their Phenotype
        Data.

    Returns
    -------
    pd.DataFrame
        DataFrame with both Genotype and Phenotype Data per individual.
    """
    # Get all individuals in the Phenotype DataFrame.
    individuals = field_phn["Pedigree"].values
    # Get all individuals in the Genotype DataFrame.
    gen_individuals = gen_df.index.values

    field_gen_phn = field_phn.copy()

    # Retrieve data for each feature in the Genotype Data.
    for variable in gen_df.columns.values:

        var_rows = []

        # Retrieve Genotype Data for each individual in the Phenotype DataFrame.
        for individual in individuals:

            # Check if we have Genotype Data for the current individual.
            if individual in gen_individuals:
                var_rows.append(gen_df.loc[individual, variable])
            # If there is no dat available, fil with "NaN".
            else:
                var_rows.append(np.nan)

        # Append data to overall DataFrame.
        var_gen = pd.DataFrame(var_rows, columns = [variable])
        field_gen_phn = pd.concat([field_gen_phn, var_gen], axis = 1)

    return field_gen_phn


def temperature_data(field_gen_phn, tmp_data, col_prefix):
    """
    Populates Temperature Data for all individuals in a target field.

    Parameters
    ----------
    field_gen_phn : pd.DataFrame
        DataFrame containing Genotype and Phenotype Data per individual.
    tmp_data : pd.DataFrame
        DataFrame containing Temperature Data for the target field.
    col_prefix : str
        Prefix to prepend to column names containing the Temperature Data.

    Returns
    -------
    pd.DataFrame
        DataFrame with Genotype, Phenotype, and Temperature Data for the target
        field.
    """
    # Create column names.
    tmp_cols = []
    for col_num in range(0, tmp_data.size):
        tmp_cols.append(col_prefix + str(col_num + 1).zfill(2))

    tmp_rows = []

    # "Populate" Weather Data for each individual in the field.
    for i in range(0, field_gen_phn.shape[0]):
        tmp_rows.append(tmp_data)

    # Append Weather Data to overall DataFrame.
    field_tmp = pd.DataFrame(tmp_rows, columns = tmp_cols)
    full_df = pd.concat([field_gen_phn, field_tmp], axis = 1).copy()

    return full_df


def read_file(file_entry):
    """
    Parses an entry from a Master File and loads the specified data into a
    corresponding DataFrame.

    Parameters
    ----------
    file_entry : pd.DataFrame
        File entry from a Master File.
    """
    parameters = file_entry.split(",")

    # Load data from a .csv file.
    if parameters[1] == "CSV":
        return pd.read_csv(parameters[2].split()[0])

    # Load Latent Dimensions from the popVAE Variational Autoencoder.
    elif parameters[1] == "LD":
        ld_df = pd.read_csv(parameters[2].split()[0], delimiter = "\t")
        ld_df = ld_df[["mean1", "mean2", "sampleID"]]
        ld_df.rename(columns = {
            "mean1": "LD_01", "mean2": "LD_02", "sampleID": "Pedigree"}
            , inplace = True)
        ld_df.set_index("Pedigree", inplace = True)
        return ld_df

    # Load data from a .h5 file.
    elif parameters[1] == "H5":
        file_path = parameters[2].split()
        return pd.read_hdf(file_path[0], file_path[1])

    # Retrieve files from a directory.
    elif parameters[1] == "DIR":
        tmp_dir = parameters[2].split()[0]
        fields = os.listdir(tmp_dir)
        return fields, tmp_dir


def main():
    global status, current, total

    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "-mf", "--master_file",
        type = str,
        help = "Path to .txt file containing dataset specifications.",
        required = True)
    parser.add_argument(
        "-pt", "--phenotype_traits",
        type = str,
        help = "Phenotype traits to be included in the master dataset, "
               "separated by commas.",
        required = True)
    parser.add_argument(
        "-md", "--master_directory",
        type = str,
        help = "Path to directory that will store generated master datasets.",
        required = True)
    parser.add_argument(
        "-fc", "--file_code",
        type = str,
        help = "Versioning code to be appended to filename.",
        required = True)

    args = parser.parse_args()

    print("\n")

    phn_df = pd.DataFrame()
    gen_dfs = []
    tmp_fields = []
    tmp_dir = ""

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Loading files into respective DataFrames.",))
    loading.start()

    with open(args.master_file, "r") as master_file:
        for line in master_file:
            if line.startswith("PHN"):
                phn_df = read_file(line)
            elif line.startswith("GEN"):
                gen_dfs.append(read_file(line))
            elif line.startswith("TMP"):
                tmp_fields, tmp_dir = read_file(line)

    status = False
    loading.join()
    tick_msg("Successfully loaded file data into DataFrames.\n")

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Obtaining list of fields present in all files.",))
    loading.start()

    phn_fields = phn_df["Field-Location"].unique()
    field_list = reduce(np.intersect1d, (phn_fields, tmp_fields))
    total = field_list.size

    status = False
    loading.join()
    tick_msg("Successfully obtained list of fields present in all files.")
    info_msg("Found full data for " + str(total) + " fields.\n")

    status = True
    loading = Thread(target = anim_process,
                     args = ("Generating Master Datasets per field.",))
    loading.start()

    for field in field_list:
        tmp_dfs = []

        # Retrieve Temperature Data.
        tmp_files = os.listdir(tmp_dir + "/" + field)

        for tmp_file in tmp_files:
            tmp_file_path = tmp_dir + "/" + field + "/" + tmp_file
            tmp_dfs.append(np.loadtxt(tmp_file_path, dtype = float))

        field_df = phn_df[phn_df["Field-Location"] == field].copy()
        master_df = phenotype_data(field_df, ["Field-Location", "Pedigree"]
                                   + args.phenotype_traits.split(","))

        for gen_df in gen_dfs:
            master_df = genotype_data(gen_df, master_df)

        i = 1
        for tmp_df in tmp_dfs:
            col_prefix = ""

            if i == 1:
                col_prefix = "ATF_"
            else:
                col_prefix = "STF_"

            master_df = temperature_data(master_df, tmp_df, col_prefix)
            i += 1

        master_file_path = args.master_directory + "/" + field + "_"\
            + args.file_code + ".h5"
        master_df.to_hdf(master_file_path, "Master")

        current += 1

    status = False
    loading.join()
    tick_msg("Successfully generated Master Datasets per field.\n")

    tick_msg("Done!")


if __name__ == "__main__":
    main()
