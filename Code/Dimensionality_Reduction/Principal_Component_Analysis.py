"""
Principal Component Analysis

This script performs dimensionality reduction on Genotype Data through a
Principal Component Analysis (PCA).

**** AgAdapt Project ****
"""


import time
import argparse
import pandas as pd
import numpy as np
from itertools import cycle
from threading import Thread
from colorama import Fore, Style
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


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


def pca_engine(genotype_df, pc, do_eigenvalues, do_variance_ratio):
    """
    Performs a Principal Component Analysis on given Genotype Data. If required,
    the function will also calculate and return the eigenvalues and the
    variance ratios for the analysis.

    Parameters
    ----------
    genotype_df : pd.DataFrame
        DataFrame containing the target Genotype Data.
    pc : int
        Number of principal components to be used for the analysis.
    do_eigenvalues : bool
        "True" if the function should calculate eigenvalues, "False" otherwise.
    do_variance_ratio : bool
        "True" if the function should calculate variance ratios, "False"
        otherwise.

    Returns
    -------
    pca_df : pd.DataFrame
        DataFrame containing calculated principal components.
    eigenvalues : array
        Array containing eigenvalues, if do_eigenvalues is set to "True".
    variance_ratio : array
        Array containing variance ratios, if do_variance_ratio is set to "True".
    """
    pca = PCA(pc)
    pca_df = pd.DataFrame(data = pca.fit_transform(genotype_df))

    names_df = pd.DataFrame(data = genotype_df.index, columns = ["Name"])
    pca_df = pd.concat([names_df, pca_df], axis = "columns")
    pca_df.set_index("Name", inplace = True)

    for i in range(0, pca_df.shape[1]):
        pca_df.rename(columns = {i: "PC_" + str(i + 1).zfill(3)},
                      inplace = True)

    eigenvalues = 0
    variance_ratio = 0

    if do_eigenvalues:
        eigenvalues = pca.explained_variance_

    if do_variance_ratio:
        variance_ratio = np.cumsum(pca.explained_variance_ratio_)

    return pca_df, eigenvalues, variance_ratio


def scree_plot(pc, eigenvalues, scree):
    """
    Creates a scree plot based on eigenvalue data and saves it to a given file.

    Parameters
    ----------
    pc : int
        Number of principal components that were used to calculate the
        eigenvalues.
    eigenvalues : array
        Array containing the calculated eigenvalues.
    scree : str
        Path to .png file to save the plot.
    """
    fig, ax = plt.subplots()
    x_values = list(range(1, pc + 1))

    ax.plot(x_values, eigenvalues, "royalblue", marker = ".")
    ax.set_title("Scree Plot of PCA ─ Component Eigenvalues",
                 loc = "right", fontname = "Monospace", fontsize = 12)
    ax.set_xlabel("Component Number", fontname = "Monospace")
    ax.set_ylabel("Eigenvalue", fontname = "Monospace")

    ax.set_facecolor("whitesmoke")
    ax.grid(True, linestyle = ":")
    ax.axis([0, pc + 1, 0, np.max(eigenvalues) + 20])

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    fig.savefig(scree, dpi = 1200)


def variance_plot(pc, variance_ratio, threshold, interpolated_pc, variance):
    """
    Creates a cumulative explained variance plot based on variance ratio data
    and saves it to a given file.

    Parameters
    ----------
    pc : int
        Number of principal components that were used to calculate the
        variance ratios.
    variance_ratio : array
        Array containing the calculated variance ratio.
    threshold : float
        Target variance to be represented in the plot.
    interpolated_pc : int
        Principal components required to preserve the target threshold.
    variance : str
        Path to .png file to save the plot.
    """
    fig, ax = plt.subplots()
    x_values = list(range(1, pc + 1))

    ax.plot(x_values, variance_ratio, "royalblue", marker = ".")
    ax.set_title("Explained Variance of PCA by Component",
                 loc = "right", fontname = "Monospace", fontsize = 12)
    ax.set_xlabel("Component Number", fontname = "Monospace")
    ax.set_ylabel("Cumulative Explained Variance", fontname = "Monospace")

    ax.set_facecolor("whitesmoke")
    ax.grid(True, linestyle = ":")
    ax.axis([0, pc + 1, 0, 1.0])

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    plt.axhline(y = threshold, linewidth = 1, color = "r", alpha = 0.5)
    plt.axvline(x = interpolated_pc, linewidth = 1, color = "r", alpha = 0.5)

    fig.savefig(variance, dpi = 1200)


def mode_a(genotype_df, pc, threshold, scree, variance):
    """
    Script mode to calculate the number of principal components needed to
    preserve a target variance.

    Parameters
    ----------
    genotype_df : pd.DataFrame
        DataFrame containing the target Genotype Data.
    pc : int
        Maximum number of principal components to use for the analysis.
    threshold : float
        Target variance to be preserved.
    scree : str
        Path to .png file to save a scree plot. If the path is an empty string,
        no plot will be created.
    variance : str
        Path to .png file to save a cumulative explained variance plot. If the
        path is an empty string, no plot will be created.
    """
    global status

    info_msg("A Principal Component Analysis will be performed.")
    info_msg("A maximum number of " + str(pc) + " principal components will be "
                                                "used.\n")

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Performing a Principal Component Analysis.",))
    loading.start()

    pca_df, eigenvalues, variance_ratio = pca_engine(genotype_df, pc,
                                                     True, True)

    status = False
    loading.join()
    tick_msg("Successfully performed a Principal Component Analysis.\n")

    if scree != "":
        info_msg("A Scree Plot will be created.")

        status = True
        loading = Thread(target = anim_loading,
                         args = ("Creating Scree Plot.",))
        loading.start()

        scree_plot(pc, eigenvalues, scree)

        status = False
        loading.join()
        tick_msg("Successfully saved Scree Plot.\n")

    info_msg("Selected Variance Threshold = " + str(threshold) + "\n")

    if threshold > np.max(variance_ratio):
        info_msg("Warning - Selected variance threshold higher than variance "
                 "conserved using " + str(pc) + " principal components.")
        info_msg("Highest Conserved Variance Possible = "
                 + format(np.max(variance_ratio), ".4f"))
        info_msg("New variance threshold will be set to "
                 + format(np.max(variance_ratio), ".4f") + "\n")
        threshold = np.max(variance_ratio)

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Interpolating principal components required for "
                             "target variance threshold.",))
    loading.start()

    x_values = np.sort(np.arange(1, pc + 1, 1))
    y_values = np.array(variance_ratio)[np.argsort(x_values)]

    interpolated_pc = int(np.round(np.interp(threshold, y_values, x_values)))

    status = False
    loading.join()
    tick_msg("Successfully interpolated principal components required for "
             "target variance threshold.\n")

    info_msg("The number of principal components needed to preserve "
             + format(threshold * 100, ".1f") + "% of variance are "
             + str(interpolated_pc) + ".\n")

    if variance != "":
        info_msg("An Explained Variance Plot will be created.")

        status = True
        loading = Thread(target = anim_loading,
                         args = ("Creating Explained Variance Plot.",))
        loading.start()

        variance_plot(pc, variance_ratio, threshold, interpolated_pc, variance)

        status = False
        loading.join()
        tick_msg("Successfully saved Explained Variance Plot.\n")


def mode_b(genotype_df, pc, out, out_key):
    """
    Script mode to run a PCA based on a specified number of principal
    components.

    Parameters
    ----------
    genotype_df : pd.DataFrame
        DataFrame containing the target Genotype Data.
    pc : int
        Number of principal components to use for the analysis.
    out : str
        Path to .h5 file to save the calculated principal components.
    out_key : str
        File key for the .h5 file that will save the calculated components.
    """
    global status

    info_msg("A Principal Component Analysis will be performed.")
    info_msg(str(pc) + " principal component(s) will be used.\n")

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Performing a Principal Component Analysis.",))
    loading.start()

    pca_df, eigenvalues, variance_ratio = pca_engine(genotype_df, pc,
                                                     False, False)

    status = False
    loading.join()
    tick_msg("Successfully performed a Principal Component Analysis.\n")

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Saving calculated principal components to .h5 "
                             "file.",))
    loading.start()

    pca_df.to_hdf(out, out_key)

    status = False
    loading.join()
    tick_msg("Successfully saved calculated principal components to .h5 file.")


def main():
    global status

    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "-g", "--genotype",
        type = str,
        help = "Path to .h5 file containing target Genotype Data.",
        required = True)
    parser.add_argument(
        "-gk", "--genotype_key",
        type = str,
        help = "File key for the .h5 file containing target Genotype Data.",
        required = True)
    parser.add_argument(
        "-m", "--mode",
        type = str,
        help = "Module mode to run. Mode A will calculate the number of "
               "principal components needed to preserve a target variance. "
               "Mode B will run a PCA based on a specified number of "
               "principal components.",
        required = True)
    parser.add_argument(
        "-pc", "--principal_components",
        type = int,
        help = "For Mode A, this sets a minimum number of principal components "
               "to calculate from. For Mode B, this sets the target principal "
               "components. If no value is given, the maximum number possible "
               "will be used.",
        default = 0)
    parser.add_argument(
        "-t", "--threshold",
        type = float,
        help = "For Mode A only. Target variance to be preserved. If no value "
               "is given, 0.8 is used.",
        default = 0.8)
    parser.add_argument(
        "-s", "--scree",
        type = str,
        help = "For Mode A only. The .png file to save a scree plot of the "
               "given data. If no value is given, no scree plot will be made.",
        default = "")
    parser.add_argument(
        "-v", "--variance",
        type = str,
        help = "For Mode A only. The .png file to save a cumulative variance "
               "plot of the data. If no value is given, no variance plot will "
               "be made.",
        default = "")
    parser.add_argument(
        "-o", "--out",
        type = str,
        help = "For Mode B only. Path to .h5 file that will store calculated "
               "principal components.")
    parser.add_argument(
        "-ok", "--out_key",
        type = str,
        help = "For Mode B only. File key for the .h5 file that will store "
               "calculated principal components.")

    args = parser.parse_args()

    print("\n")

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Loading Genotype Data into a Pandas DataFrame.",))
    loading.start()

    genotype_df = pd.read_hdf(args.genotype, args.genotype_key)

    status = False
    loading.join()
    tick_msg("Successfully loaded Genotype Data into a Pandas DataFrame.")
    info_msg("Total Individuals = " + str(genotype_df.shape[0]))
    info_msg("Chromosome Positions per Individual = "
             + str(genotype_df.shape[1]) + "\n")

    pc = args.principal_components
    if pc == 0:
        pc = genotype_df.shape[0]

    if args.mode == 'A':
        info_msg("Mode A is selected.")
        mode_a(genotype_df, pc, args.threshold,
               args.scree, args.variance)

    elif args.mode == 'B':
        info_msg("Mode B is selected.")
        mode_b(genotype_df, pc, args.out, args.out_key)

    else:
        info_msg("Invalid Mode.")

    tick_msg("Done!")


if __name__ == "__main__":
    main()
