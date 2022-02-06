"""
Offspring Genotype Prediction

This script predicts Genotype Data for target offspring individuals, given the
Genotype Data of both parents and the corresponding pedigree of the offspring.

**** AgAdapt Project ****
"""


import time
import argparse
import pandas as pd
import numpy as np
from itertools import cycle
from threading import Thread
from colorama import Fore, Style


status = False
current = 0
total = 1
offspring_df = ""
genotype_df = ""


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


def predict(offspring, female_parent, male_parent):
    """
    Predicts the genotype data of a target offspring given parental data and
    the average allele per chromosome position.

    Genotype data is predicted depending on whether the combination of parental
    data and average allele fits 1 of 15 cases.

    This function will not work with missing data and has no default case.

    Parameters
    ----------
    offspring : str
        Unique pedigree or code for the target offspring.
    female_parent : str
        Unique pedigree or code for the Female Parent of the target offspring.
    male_parent : str
        Unique pedigree or code for the Male Parent of the target offspring.

    Returns
    -------
    pd.DataFrame
        Modified offspring dataframe.
    """

    global current

    conditions = [

        # ----------------------------------------------------------------------
        # Case 01 - [0]
        # Both parents are homozygous, but no SNP is present. [0]
        # Offspring should be homozygous, with no SNP. [0]
        # ----------------------------------------------------------------------

        (genotype_df.loc[female_parent] == 0) &
        (genotype_df.loc[male_parent] == 0),

        # ----------------------------------------------------------------------
        # Case 02 - [1]
        # Both parents are heterozygous. [1]
        # On average, individuals are heterozygous. [1]
        # Offspring should be heterozygous. [1]
        # ----------------------------------------------------------------------

        (genotype_df.loc[female_parent] == 1) &
        (genotype_df.loc[male_parent] == 1) &
        (genotype_df.loc["Mean"] == 1),

        # ----------------------------------------------------------------------
        # Case 03 - [2]
        # Both parents are heterozygous. [1]
        # On average, individuals are homozygous for SNP. [2]
        # Offspring should be homozygous for SNP. [2]
        # ----------------------------------------------------------------------

        (genotype_df.loc[female_parent] == 1) &
        (genotype_df.loc[male_parent] == 1) &
        (genotype_df.loc["Mean"] == 2),

        # ----------------------------------------------------------------------
        # Case 04 - [0]
        # Both parents are heterozygous. [1]
        # On average, individuals are homozygous, with no SNP. [0]
        # Offspring should be homozygous, with no SNP. [0]
        # ----------------------------------------------------------------------

        (genotype_df.loc[female_parent] == 1) &
        (genotype_df.loc[male_parent] == 1) &
        (genotype_df.loc["Mean"] == 0),

        # ----------------------------------------------------------------------
        # Case 05 - [2]
        # Both parents are homozygous, with SNP. [2]
        # Offspring should be homozygous, with SNP. [2]
        # ----------------------------------------------------------------------

        (genotype_df.loc[female_parent] == 2) &
        (genotype_df.loc[male_parent] == 2),

        # ----------------------------------------------------------------------
        # Case 06 - [0]
        # Female Parent is heterozygous. [1]
        # Male Parent is homozygous, but no SNP is present. [0]
        # On average, individuals are homozygous, with no SNP. [0]
        # Offspring should be homozygous, with no SNP. [0]
        # ----------------------------------------------------------------------

        (genotype_df.loc[female_parent] == 1) &
        (genotype_df.loc[male_parent] == 0) &
        (genotype_df.loc["Mean"] == 0),

        # ----------------------------------------------------------------------
        # Case 07 - [1]
        # Female Parent is heterozygous. [1]
        # Male Parent is homozygous, but no SNP is present. [0]
        # On average, individuals are heterozygous. [1]
        # Offspring should be heterozygous. [1]
        # ----------------------------------------------------------------------

        (genotype_df.loc[female_parent] == 1) &
        (genotype_df.loc[male_parent] == 0) &
        (genotype_df.loc["Mean"] == 1),

        # ----------------------------------------------------------------------
        # Case 08 - [2]
        # Female Parent is heterozygous. [1]
        # Male Parent is homozygous, with SNP. [2]
        # On average, individuals are homozygous, with SNP. [2]
        # Offspring should be homozygous, with SNP. [2]
        # ----------------------------------------------------------------------

        (genotype_df.loc[female_parent] == 1) &
        (genotype_df.loc[male_parent] == 2) &
        (genotype_df.loc["Mean"] == 2),

        # ----------------------------------------------------------------------
        # Case 09 - [1]
        # Female Parent is heterozygous. [1]
        # Male Parent is homozygous, with SNP. [2]
        # On average, individuals are heterozygous. [1]
        # Offspring should be heterozygous. [1]
        # ----------------------------------------------------------------------

        (genotype_df.loc[female_parent] == 1) &
        (genotype_df.loc[male_parent] == 2) &
        (genotype_df.loc["Mean"] == 1),

        # ----------------------------------------------------------------------
        # Case 10 - [1]
        # Female Parent is homozygous, with SNP. [2]
        # Male Parent is homozygous, but no SNP is present. [0]
        # Offspring should be heterozygous. [1]
        # ----------------------------------------------------------------------

        (genotype_df.loc[female_parent] == 2) &
        (genotype_df.loc[male_parent] == 0),

        # ----------------------------------------------------------------------
        # Case 11 - [1]
        # Female Parent is homozygous, but no SNP is present. [0]
        # Male Parent is homozygous, with SNP. [2]
        # Offspring should be heterozygous. [1]
        # ----------------------------------------------------------------------

        (genotype_df.loc[female_parent] == 0) &
        (genotype_df.loc[male_parent] == 2),

        # ----------------------------------------------------------------------
        # Case 12 - [0]
        # Female Parent is homozygous, but no SNP is present. [0]
        # Male Parent is heterozygous. [1]
        # On average, individuals are homozygous, with no SNP. [0]
        # Offspring should be homozygous, with no SNP. [0]
        # ----------------------------------------------------------------------

        (genotype_df.loc[female_parent] == 0) &
        (genotype_df.loc[male_parent] == 1) &
        (genotype_df.loc["Mean"] == 0),

        # ----------------------------------------------------------------------
        # Case 13 - [1]
        # Female Parent is homozygous, but no SNP is present. [0]
        # Male Parent is heterozygous. [1]
        # On average, individuals are heterozygous. [1]
        # Offspring should be heterozygous. [1]
        # ----------------------------------------------------------------------

        (genotype_df.loc[female_parent] == 0) &
        (genotype_df.loc[male_parent] == 1) &
        (genotype_df.loc["Mean"] == 1),

        # ----------------------------------------------------------------------
        # Case 14 - [2]
        # Female Parent is homozygous, with SNP. [2]
        # Male Parent is heterozygous. [1]
        # On average, individuals are homozygous, with SNP. [2]
        # Offspring should be homozygous, with SNP. [2]
        # ----------------------------------------------------------------------

        (genotype_df.loc[female_parent] == 2) &
        (genotype_df.loc[male_parent] == 1) &
        (genotype_df.loc["Mean"] == 2),

        # ----------------------------------------------------------------------
        # Case 15 - [1]
        # Female Parent is homozygous, with SNP. [2]
        # Male Parent is heterozygous. [1]
        # On average, individuals are heterozygous. [1]
        # Offspring should be heterozygous. [1]
        # ----------------------------------------------------------------------

        (genotype_df.loc[female_parent] == 2) &
        (genotype_df.loc[male_parent] == 1) &
        (genotype_df.loc["Mean"] == 1)]

    predicted_genotype = [0, 1, 2, 0, 2, 0, 1, 2, 1, 1, 1, 0, 1, 2, 1]

    offspring_df.loc[offspring] = np.select(conditions, predicted_genotype)

    current += 1


def main():
    global status, current, total, genotype_df, offspring_df

    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "-g", "--genotype",
        type = str,
        help = "Path to .h5 file containing Parental Genotype Data.",
        required = True)
    parser.add_argument(
        "-gk", "--genotype_key",
        type = str,
        help = "File key for the .h5 file containing Parental Genotype Data.",
        required = True)
    parser.add_argument(
        "-c", "--codes",
        type = str,
        help = "Path to .csv file containing offspring pedigrees or codes.",
        required = True)
    parser.add_argument(
        "-o", "--offspring",
        type = str,
        help = "Path to .h5 file that will store predicted offspring data.",
        required = True)
    parser.add_argument(
        "-ok", "--offspring_key",
        type = str,
        help = "File key for the .h5 file that will store predicted offspring "
               "data.",
        required = True)

    args = parser.parse_args()

    print("\n")

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Loading Parental Genotype Data into a Pandas "
                             "DataFrame.",))
    loading.start()

    genotype_df = pd.read_hdf(args.genotype, args.genotype_key)

    status = False
    loading.join()
    tick_msg("Successfully loaded Parental Genotype Data into a Pandas "
             "DataFrame.")
    info_msg("Total Individuals = " + str(genotype_df.shape[0]))
    info_msg("Chromosome Positions per Individual = "
             + str(genotype_df.shape[1]) + "\n")

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Loading offspring pedigrees/codes into a Pandas "
                             "DataFrame.",))
    loading.start()

    codes_df = pd.read_csv(args.codes)

    status = False
    loading.join()
    tick_msg("Successfully loaded offspring pedigrees/codes into a Pandas "
             "DataFrame.")
    info_msg("Found " + str(codes_df.shape[0]) + " offspring individuals.\n")

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Determining average allele per position in "
                             "Parental Genotype Data.",))
    loading.start()

    mean_df = pd.DataFrame([genotype_df.mean(axis = 0)], index = ["Mean"])
    mean_df = np.round(mean_df).astype(int)

    genotype_df = pd.concat([genotype_df, mean_df], axis = 0)

    status = False
    loading.join()
    tick_msg("Successfully determined average allele per position in "
             "Parental Genotype Data.\n")

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Creating empty DataFrame to store predicted "
                             "offspring data.",))
    loading.start()

    offspring_df = pd.DataFrame(columns = genotype_df.columns.values)

    status = False
    loading.join()
    tick_msg("Successfully created empty DataFrame to store predicted "
             "offspring data.\n")

    status = True
    loading = Thread(target = anim_process,
                     args = ("Predicting genotype data for each offspring.",))
    loading.start()

    total = codes_df.shape[0]

    codes_df.apply(lambda row: predict(row["Pedigree"], row["Female Parent"],
                                       row["Male Parent"]), axis = 1)

    status = False
    loading.join()
    tick_msg("Successfully predicted genotype data for each offspring.")
    info_msg("Predicted " + str(offspring_df.shape[0]) + " individuals.\n")

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Saving predicted offspring data to .h5 file.",))
    loading.start()

    offspring_df = offspring_df.astype(int)
    offspring_df.to_hdf(args.offspring, args.offspring_key)

    status = False
    loading.join()
    tick_msg("Successfully saved imputed data to .h5 file.\n")

    tick_msg("Done!")


if __name__ == "__main__":
    main()
