"""
Genotype Data Cleaning

This script filters individuals and chromosome positions with a target amount
of missing data.

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
        "-gf", "--genotype_files",
        type = str,
        help = "Path to .txt file containing file paths and file keys for "
               "target Genotype Data .h5 files.",
        required = True)
    parser.add_argument(
        "-it", "--individual_threshold",
        type = float,
        help = "Filtering threshold for individuals. Any individual with an "
               "amount of data lower than the threshold will be dropped.",
        required = True)
    parser.add_argument(
        "-pt", "--position_threshold",
        type = float,
        help = "Filtering threshold for chromosome positions. Any position "
               "an amount of data lower than the threshold will be dropped.",
        required = True)
    parser.add_argument(
        "-of", "--output_files",
        type = str,
        help = "Path to .txt file containing file paths and file keys for "
               ".h5 files that will store clean Genotype Data.",
        required = True)

    args = parser.parse_args()

    print("\n")

    gen_fl = open(args.genotype_files, "r")
    out_fl = open(args.output_files, "r")

    gen_files = gen_fl.readlines()
    out_files = out_fl.readlines()

    gen_fl.close()
    out_fl.close()

    chr_list = []
    chr_keys = []

    for gen_file in gen_files:
        gen_file = gen_file.strip()
        chr_list.append(gen_file.split(" ")[0])
        chr_keys.append(gen_file.split(" ")[1])

    out_list = []
    out_keys = []

    for out_file in out_files:
        out_file = out_file.strip()
        out_list.append(out_file.split(" ")[0])
        out_keys.append(out_file.split(" ")[1])

    info_msg("Found Genotype Data for " + str(len(chr_list))
             + " chromosomes.\n")

    missing_counts = pd.DataFrame()
    position_count = 0

    for i in range(0, len(chr_list)):

        status = True
        loading = Thread(target = anim_loading,
                         args = ("Identifying missing Genotype Data for "
                                 "Chromosome " + str(i + 1) + ".",))
        loading.start()

        chr_df = pd.read_hdf(chr_list[i], chr_keys[i])

        missing_counts = pd.concat([missing_counts,
                                    chr_df.isna().sum(axis = 1)], axis = 1)
        missing_counts.columns = [*missing_counts.columns[:-1],
                                  "CHR_" + str(i + 1).zfill(2)]

        chr_pos = chr_df.shape[1]
        position_count += chr_pos

        status = False
        loading.join()
        tick_msg("Successfully identified missing Genotype Data for "
                 "Chromosome " + str(i + 1) + ". [P = " + str(chr_pos) + "]")

    print("\n")

    individual_count = missing_counts.shape[0]

    info_msg("Total Individuals = " + str(individual_count))
    info_msg("Chromosome Positions per Individual = "
             + str(position_count) + "\n")

    info_msg("Target Threshold for Individuals = "
             + str(args.individual_threshold))
    info_msg("Target Threshold for Chromosome Positions = "
             + str(args.position_threshold) + "\n")

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Filtering individuals.",))
    loading.start()

    min_positions = int(np.ceil(args.individual_threshold * position_count))
    missing_counts = pd.DataFrame(missing_counts.sum(axis = 1),
                                  columns = ["Total"])
    filtered_df = missing_counts[missing_counts["Total"] <= min_positions]

    status = False
    loading.join()
    tick_msg("Successfully filtered individuals.\n")

    min_individuals = int(np.ceil(args.position_threshold *
                                  individual_count))

    new_position_count = 0

    for i in range(0, len(chr_list)):

        status = True
        loading = Thread(target = anim_loading,
                         args = ("Applying filtering to and saving "
                                 "Chromosome " + str(i + 1) + ".",))
        loading.start()

        chr_df = pd.read_hdf(chr_list[i], chr_keys[i])
        chr_df = chr_df[chr_df.index.isin(filtered_df.index)]

        chr_df = chr_df.dropna(axis = 1, thresh = min_individuals)

        chr_pos = chr_df.shape[1]
        new_position_count += chr_pos

        chr_df.to_hdf(out_list[i], out_keys[i])

        status = False
        loading.join()
        tick_msg("Successfully applied filtering to and saved "
                 "Chromosome " + str(i + 1) + ". [P = " + str(chr_pos) + "]")

    print("\n")

    info_msg("Number of individuals is now " + str(filtered_df.shape[0]) + ".")
    info_msg(str(individual_count - filtered_df.shape[0]) + " individuals "
             + "were dropped.\n")

    info_msg("Number of chromosome positions is now "
             + str(new_position_count) + ".")
    info_msg(str(position_count - new_position_count) + " positions "
             + "were dropped.\n")

    tick_msg("Done!")


if __name__ == "__main__":
    main()
