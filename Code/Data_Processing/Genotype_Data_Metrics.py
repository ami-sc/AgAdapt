"""
Genotype Data Metrics

This script calculates general metrics for each chromosome position of a set
of individuals.

**** AgAdapt Project ****
"""


import time
import argparse
import numpy as np
import pandas as pd
from itertools import cycle
from threading import Thread
from colorama import Fore, Style


status = False


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
            time.sleep(0.3)
        else:
            break


def metrics(chr_position, genotype_df, metrics_df):
    """
    Calculates metrics for a given chromosome position.

    Parameters
    ----------
    chr_position : str
        Chromosome position from which metrics will be calculated.
    genotype_df : pd.DataFrame
        DataFrame containing Genotype Data for each individual.
    metrics_df : pd.DataFrame
        DataFrame that will store calculated metrics.

    Returns
    -------
    pd.DataFrame
        Modified metrics DataFrame.
    """
    homozygous_mt = (genotype_df[chr_position] == 2).sum(axis = 0)
    heterozygous = (genotype_df[chr_position] == 1).sum(axis = 0)
    homozygous_wt = (genotype_df[chr_position] == 0).sum(axis = 0)
    total_ind = homozygous_mt + heterozygous + homozygous_wt

    per_homozygous_mt = (homozygous_mt / total_ind) * 100
    per_heterozygous = (heterozygous / total_ind) * 100
    per_homozygous_wt = (homozygous_wt / total_ind) * 100

    observed_mt = (homozygous_mt * 2) + heterozygous
    observed_wt = (homozygous_wt * 2) + heterozygous

    frequency_mt = (observed_mt / (total_ind * 2)) * 100
    frequency_wt = (observed_wt / (total_ind * 2)) * 100

    metrics_df.loc["Total Individuals (#)", chr_position] = total_ind

    metrics_df.loc["Homozygous Mutant (#)", chr_position] = homozygous_mt
    metrics_df.loc["Heterozygous (#)", chr_position] = heterozygous
    metrics_df.loc["Homozygous WildType (#)", chr_position] = homozygous_wt

    metrics_df.loc["Homozygous Mutant (%)", chr_position] = per_homozygous_mt
    metrics_df.loc["Heterozygous (%)", chr_position] = per_heterozygous
    metrics_df.loc["Homozygous WildType (%)", chr_position] = per_homozygous_wt

    metrics_df.loc["Observed Mutant Copies (#)", chr_position] = observed_mt
    metrics_df.loc["Observed WildType Copies (#)", chr_position] = observed_wt

    metrics_df.loc["Frequency of Mutant (%)", chr_position] = frequency_mt
    metrics_df.loc["Frequency of WildType (%)", chr_position] = frequency_wt

    return metrics_df


def main():
    global status

    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "-g", "--genotype",
        type = str,
        help = "Path to .h5 file containing Genotype Data.",
        required = True)
    parser.add_argument(
        "-k", "--key",
        type = str,
        help = "File key for the .h5 file containing Genotype Data.",
        required = True)
    parser.add_argument(
        "-m", "--metrics",
        type = str,
        help = "Path to .csv file that will store calculated metrics.",
        required = True)

    args = parser.parse_args()

    print("\n")

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Loading .h5 file data into a Pandas DataFrame.",))
    loading.start()

    genotype_df = pd.read_hdf(args.genotype, args.key)

    status = False
    loading.join()
    tick_msg("Successfully loaded .h5 file data into a Pandas DataFrame.")

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Discarding BLANK samples.",))
    loading.start()

    genotype_df = genotype_df[~genotype_df.index.str.contains("BLANK")].copy()

    status = False
    loading.join()
    tick_msg("Successfully discarded BLANK samples.")

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Creating DataFrame to store metrics.",))
    loading.start()

    position_list = genotype_df.columns.values
    metrics_df = pd.DataFrame(columns = position_list)

    metrics_df.loc["Total Individuals (#)"] = np.nan

    metrics_df.loc["Homozygous Mutant (#)"] = np.nan
    metrics_df.loc["Heterozygous (#)"] = np.nan
    metrics_df.loc["Homozygous WildType (#)"] = np.nan

    metrics_df.loc["Homozygous Mutant (%)"] = np.nan
    metrics_df.loc["Heterozygous (%)"] = np.nan
    metrics_df.loc["Homozygous WildType (%)"] = np.nan

    metrics_df.loc["Observed Mutant Copies (#)"] = np.nan
    metrics_df.loc["Observed WildType Copies (#)"] = np.nan

    metrics_df.loc["Frequency of Mutant (%)"] = np.nan
    metrics_df.loc["Frequency of WildType (%)"] = np.nan

    status = False
    loading.join()
    tick_msg("Successfully created DataFrame to store metrics.")

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Calculating metrics for each position.",))
    loading.start()

    for position in position_list:
        metrics_df = metrics(position, genotype_df, metrics_df)

    status = False
    loading.join()
    tick_msg("Successfully calculated metrics for each position.")

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Saving metrics to .csv file.",))
    loading.start()

    metrics_df = metrics_df.astype(float)
    metrics_df.to_csv(args.metrics)

    status = False
    loading.join()
    tick_msg("Successfully saved metrics to .csv file.")

    tick_msg("Done!")


if __name__ == "__main__":
    main()
