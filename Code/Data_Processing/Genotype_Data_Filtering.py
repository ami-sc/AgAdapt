"""
Genotype Data Filtering

This script filters SNP data according to a provided metric and a specified
threshold.

**** AgAdapt Project ****
"""


import time
import argparse
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
            time.sleep(0.4)
        else:
            break


def main():
    global status

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
        "-m", "--metrics",
        type = str,
        help = "Path to .csv file containing metrics of given Genotype data.",
        required = True)
    parser.add_argument(
        "-f", "--filter",
        type = str,
        help = "Metric to be used for filtering.",
        required = True)
    parser.add_argument(
        "-t", "--threshold",
        type = float,
        help = "Threshold for given filtering metric.",
        required = True)
    parser.add_argument(
        "-o", "--out",
        type = str,
        help = "Path to .h5 file that will store filtered Genotype data.",
        required = True)
    parser.add_argument(
        "-ok", "--out_key",
        type = str,
        help = "File key for the .h5 file that will store filtered Genotype "
               "data.",
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
    info_msg("Total Individuals = " + str(genotype_df.shape[0]))

    original_positions = genotype_df.shape[1]
    info_msg("Chromosome Positions per Individual = "
             + str(original_positions) + "\n")

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Loading metrics data into a Pandas DataFrame.",))
    loading.start()

    metrics_df = pd.read_csv(args.metrics, index_col = 0)

    status = False
    loading.join()
    tick_msg("Successfully loaded metrics data into a Pandas DataFrame.")
    info_msg("Found " + str(metrics_df.shape[0]) + " metrics.\n")

    info_msg("Target Metric = " + args.filter)
    info_msg("Target Threshold = " + str(args.threshold))

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Concatenating target metric to Genotype data.",))
    loading.start()

    genotype_df = pd.concat([genotype_df, metrics_df.loc[[args.filter]]],
                            axis = 0)

    status = False
    loading.join()
    tick_msg("Successfully concatenated target metric to Genotype data.\n")

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Filtering Genotype data.",))
    loading.start()

    genotype_df = genotype_df.loc[:, (genotype_df.loc[[args.filter]] >=
                                      args.threshold).any()]

    status = False
    loading.join()
    tick_msg("Successfully filtered Genotype data.\n")

    total_positions = genotype_df.shape[1]
    discarded_positions = original_positions - total_positions
    info_msg(str(discarded_positions) + " chromosome positions discarded.")
    info_msg("Total positions are now " + str(total_positions) + ".\n")

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Removing target metric from Genotype data.",))
    loading.start()

    genotype_df.drop(args.filter, inplace = True)

    status = False
    loading.join()
    tick_msg("Successfully removed target metric from Genotype data.\n")

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Saving imputed data to .h5 file.",))
    loading.start()

    genotype_df = genotype_df.astype(int)
    genotype_df.to_hdf(args.out, args.out_key)

    status = False
    loading.join()
    tick_msg("Successfully saved imputed data to .h5 file.\n")

    tick_msg("Done!")


if __name__ == "__main__":
    main()
