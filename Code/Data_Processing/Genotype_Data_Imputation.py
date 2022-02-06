"""
Genotype Data Imputation

This script imputes missing Genotype Data, per chromosome position, with the
most frequently observed allele.

**** AgAdapt Project ****
"""


import time
import argparse
import pandas as pd
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


def impute(chr_position, genotype_df):
    """
    Imputes the missing genotypes in a given position.

    The genotype will be imputed according to the most frequent observed allele
    in the target position:

    - Case 1: The most frequent observed allele is Homozygous Mutant ('2').
    - Case 2: The most frequent observed allele is Heterozygous ('1').
    - Case 3: The most frequent observed allele is Homozygous WildType ('0').

    If more than one of these conditions is true or data for all individuals
    in a position is missing, the function will default to Heterozygous ('1').

    Parameters
    ----------
    chr_position : str
        The chromosome position to be imputed.
    genotype_df : pd.DataFrame
        Genotype dataframe containing genetic data for each individual.

    Returns
    -------
    pd.DataFrame
        Modified genotype dataframe.
    """
    homozygous_mt = (genotype_df[chr_position] == 2).sum(axis = 0)
    heterozygous = (genotype_df[chr_position] == 1).sum(axis = 0)
    homozygous_wt = (genotype_df[chr_position] == 0).sum(axis = 0)

    # Case 1 - Homozygous Mutant
    if (homozygous_mt > heterozygous) and (homozygous_mt > homozygous_wt):
        genotype_df[chr_position].fillna(2, inplace = True)

    # Case 2 - Heterozygous
    elif (heterozygous > homozygous_mt) and (heterozygous > homozygous_wt):
        genotype_df[chr_position].fillna(1, inplace=True)

    # Case 3 - Homozygous WildType
    elif (homozygous_wt > heterozygous) and (homozygous_wt > homozygous_mt):
        genotype_df[chr_position].fillna(0, inplace = True)

    # Default - Heterozygous
    else:
        genotype_df[chr_position].fillna(1, inplace=True)

    return genotype_df


def main():
    global status, current, total

    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "-g", "--genotype",
        type = str,
        help = "Path to .h5 file containing Genotype Data.",
        required = True)
    parser.add_argument(
        "-gk", "--genotype_key",
        type = str,
        help = "File key for the .h5 file containing Genotype Data.",
        required = True)
    parser.add_argument(
        "-i", "--imputed",
        type = str,
        help = "Path to .h5 file that will store imputed data.",
        required = True)
    parser.add_argument(
        "-ik", "--imputed_key",
        type = str,
        help = "File key for the .h5 file that will store imputed data.",
        required = True)

    args = parser.parse_args()

    print("\n")

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Loading .h5 file data into a Pandas DataFrame.",))
    loading.start()

    genotype_df = pd.read_hdf(args.genotype, args.genotype_key)

    status = False
    loading.join()
    tick_msg("Successfully loaded .h5 file data into a Pandas DataFrame.")
    info_msg("Total Individuals = " + str(genotype_df.shape[0]) + "\n")

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Discarding BLANK samples.",))
    loading.start()

    discarded_blanks = genotype_df.index.str.contains("BLANK").sum()
    genotype_df = genotype_df[~genotype_df.index.str.contains("BLANK")].copy()

    status = False
    loading.join()
    tick_msg("Discarded " + str(discarded_blanks) + " BLANK samples.")
    info_msg("Total Individuals (No BLANKS) = " + str(genotype_df.shape[0])
             + "\n")

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Retrieving positions from dataframe.",))
    loading.start()

    imputed_count = genotype_df.isnull().sum().sum()
    position_list = genotype_df.columns.values
    total = len(position_list)

    status = False
    loading.join()
    tick_msg("Successfully retrieved positions from dataframe.\n")

    status = True
    loading = Thread(target = anim_process,
                     args = ("Imputing missing data for each position.",))
    loading.start()

    for position in position_list:
        genotype_df = impute(position, genotype_df)
        current += 1

    status = False
    loading.join()
    tick_msg("Successfully imputed missing data for each position.")
    info_msg(str(imputed_count) + " sites were imputed.\n")

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Saving imputed data to .h5 file.",))
    loading.start()

    genotype_df = genotype_df.astype(int)
    genotype_df.to_hdf(args.imputed, args.imputed_key)

    status = False
    loading.join()
    tick_msg("Successfully saved imputed data to .h5 file.\n")

    tick_msg("Done!")


if __name__ == "__main__":
    main()
