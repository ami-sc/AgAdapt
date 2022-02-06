"""
Temperature Data Plots

This script generates per-field monthly and yearly plots for given air and soil
temperature data from the 2017 G2F project measurements. Additionally, a summary
file containing basic statistics of given field data will be generated.

**** AgAdapt Project ****
"""

import os
import time
import argparse
import pandas as pd
import numpy as np
import calendar
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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


def plot_year(field_df, field_name, month_list, plt_type, path):
    """
    Generates two yearly plots of air and soil temperature data, respectively.

    Parameters
    ----------
    field_df : pd.DataFrame
        DataFrame containing the air and soil temperature data of a target field
        to be plotted.
    field_name : str
        Name of the target field.
    month_list : ndarray
        Array containing the months present in the given DataFrame, in numerical
        format.
    plt_type : str
        If "Scatter", a scatter plot will be generated. If "Linear", a linear
        plot will be generated. Any other option will cause an error.
    path : str
        Directory path where the generated plots will be saved.
    """
    measurements = ["Air", "Soil"]

    for measurement in measurements:

        y = ""
        plot_color = ""
        month_names = []
        locations = []

        # Create plot.
        figure = plt.figure(figsize = (20, 5))
        year_plot = figure.add_subplot(1, 1, 1)

        plot_title = field_name + " - 2017 " + measurement + " Temperature"

        year_plot.set_title(plot_title, loc = "right", fontsize = 18)
        year_plot.set_ylabel(measurement + " Temperature (°C)", fontsize = 12)
        year_plot.set_facecolor("#F5F5F5")
        year_plot.grid(True, linestyle = ':')

        year_plot.axes.get_xaxis().set_ticks([])
        year_plot.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        year_plot.axis(xmin = np.min(month_list), xmax = np.max(month_list) + 1)

        # Plot data per-month.
        for month in month_list:

            # Plot line to separate months.
            year_plot.axvline(x = int(month), color = "grey", linestyle = ':')

            # Retrieve data for target month.
            month_df = field_df[field_df["Month"] == month]

            # Get number of days in given month.
            days = calendar.monthrange(2017, month)[1]

            if measurement == "Air":
                y = month_df["Temperature [C]"].to_numpy()
                plot_color = "#0091FF"

            elif measurement == "Soil":
                y = month_df["Soil Temperature [C]"].to_numpy()
                plot_color = "red"

            x = month_df["Month"] + ((month_df["Day"] - 1) / days)\
                + (month_df["Decimal Time"] / (days * 24))

            if plt_type == "Scatter":
                year_plot.scatter(x, y, s = 2, c = plot_color)

            elif plt_type == "Linear":
                year_plot.plot(x, y, c = plot_color)

            # Get month name and label location.
            month_names.append(calendar.month_name[month])
            locations.append((int(month) + int(month) + 1) / 2)

        year_plot.xaxis.set_minor_locator(ticker.FixedLocator(locations))
        year_plot.xaxis.set_minor_formatter(ticker.FixedFormatter(month_names))
        plt.setp(year_plot.xaxis.get_minorticklabels(), size = 10)
        year_plot.tick_params(axis = 'x', which = "minor", pad = 12, length = 0)

        # Construct file name and save plot.
        plot_file = path + '/' + "2017_" + measurement \
            + "_Temperature_" + plt_type + "_Plot.png"

        plt.savefig(plot_file, dpi = 600)
        plt.close()


def plot_month(month_df, field_name, month, plt_type, path):
    """
    Generates monthly plots of air and soil temperature data, respectively.

    Parameters
    ----------
    month_df : pd.DataFrame
        DataFrame containing the air and soil temperature data of a target month
        to be plotted.
    field_name : str
        Name of the target field.
    month : int
        Month being plotted, in numerical format.
    plt_type : str
        If "Scatter", a scatter plot will be generated. If "Linear", a linear
        plot will be generated. Any other option will cause an error.
    path : str
        Directory path where the generated plots will be saved.
    """
    measurements = ["Air", "Soil"]
    y = ""

    for measurement in measurements:

        plot_color = ""

        if measurement == "Air":
            y = month_df["Temperature [C]"].to_numpy()
            plot_color = "#0091FF"

        elif measurement == "Soil":
            y = month_df["Soil Temperature [C]"].to_numpy()
            plot_color = "red"

        month_name = calendar.month_name[month]
        days = calendar.monthrange(2017, month)[1]

        figure = plt.figure(figsize = (20, 5))
        month_plot = figure.add_subplot(1, 1, 1)

        plot_title = field_name + " - " + month_name + " 2017 " \
            + measurement + " Temperature"

        month_plot.set_title(plot_title, loc = "right", fontsize = 18)
        month_plot.set_xlabel("Day", fontsize = 12)
        month_plot.set_ylabel(measurement + " Temperature (°C)", fontsize = 12)
        month_plot.set_facecolor("#F5F5F5")
        month_plot.grid(True, linestyle = ':')

        month_plot.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        month_plot.axis(xmin = 1, xmax = days + 1)
        month_plot.xaxis.set_major_locator(ticker.MultipleLocator(1))

        x = month_df["Day"] + (month_df["Decimal Time"] / 24)

        if plt_type == "Scatter":
            month_plot.scatter(x, y, s = 2, color = plot_color)

        elif plt_type == "Linear":
            month_plot.plot(x, y, color = plot_color)

        # Construct file name and save plot.
        plot_file = path + '/' + "{:02d}".format(month) + "_2017_" \
            + measurement + "_Temperature_" + plt_type + "_Plot.png"

        plt.savefig(plot_file, dpi = 600)
        plt.close()


def field_summary(field_df, field_name, path):
    """
    Generates a markdown file containing temperature statistics of a given
    field, along with generated air and soil temperature plots.

    Parameters
    ----------
    field_df : pd.DataFrame
        DataFrame containing the temperature data of a target field.
    field_name : str
        Name of the target field.
    path : str
        Directory path where the generated summary will be saved.
    """
    summary_path = path + '/' + "README.md"

    measurements = field_df.shape[0]
    missing_air = np.count_nonzero(field_df["Temperature [C]"].isnull())
    missing_soil = np.count_nonzero(field_df["Soil Temperature [C]"].isnull())

    pm_air = (missing_air / measurements) * 100
    pm_soil = (missing_soil / measurements) * 100

    max_air = field_df["Temperature [C]"].max()
    min_air = field_df["Temperature [C]"].min()
    max_soil = field_df["Soil Temperature [C]"].max()
    min_soil = field_df["Soil Temperature [C]"].min()

    mean_air = field_df["Temperature [C]"].mean()
    mean_soil = field_df["Soil Temperature [C]"].mean()

    std_air = field_df["Temperature [C]"].std()
    std_soil = field_df["Soil Temperature [C]"].std()

    months = field_df["Month"].unique()

    table_headers = ["Days Measured [#]", "Measurements [#]", "Max T [C]",
                     "Min T [C]", "Avg T [C]", "Std T [C]", "Missing [C]",
                     "Missing [%]"]

    air_summary = pd.DataFrame(columns = table_headers)
    soil_summary = pd.DataFrame(columns = table_headers)

    air_plot_paths = []
    soil_plot_paths = []

    for month in months:

        air_statistics = []
        soil_statistics = []

        month_df = field_df[field_df["Month"] == month]

        days_measured = month_df["Day"].nunique()
        num_measurements = month_df.shape[0]

        air_m = np.count_nonzero(month_df["Temperature [C]"].isnull())
        soil_m = np.count_nonzero(month_df["Soil Temperature [C]"].isnull())

        air_statistics.append(days_measured)
        air_statistics.append(num_measurements)
        air_statistics.append(
            "{:.2f}".format(month_df["Temperature [C]"].max()))
        air_statistics.append(
            "{:.2f}".format(month_df["Temperature [C]"].min()))
        air_statistics.append(
            "{:.2f}".format(month_df["Temperature [C]"].mean()))
        air_statistics.append(
            "{:.2f}".format(month_df["Temperature [C]"].std()))
        air_statistics.append(air_m)
        air_statistics.append(
            "{:.2f}".format((air_m / num_measurements) * 100))

        soil_statistics.append(days_measured)
        soil_statistics.append(num_measurements)
        soil_statistics.append(
            "{:.2f}".format(month_df["Soil Temperature [C]"].max()))
        soil_statistics.append(
            "{:.2f}".format(month_df["Soil Temperature [C]"].min()))
        soil_statistics.append(
            "{:.2f}".format(month_df["Soil Temperature [C]"].mean()))
        soil_statistics.append(
            "{:.2f}".format(month_df["Soil Temperature [C]"].std()))
        soil_statistics.append(soil_m)
        soil_statistics.append(
            "{:.2f}".format((soil_m / num_measurements) * 100))

        month_name = calendar.month_name[month]

        air_summary.loc[month_name] = air_statistics
        soil_summary.loc[month_name] = soil_statistics

        # Generate plot paths.
        air_plot_paths.append("{:02d}".format(month)
                              + "_2017_Air_Temperature_Scatter_Plot.png")
        soil_plot_paths.append("{:02d}".format(month)
                               + "_2017_Soil_Temperature_Scatter_Plot.png")

    summary = open(summary_path, "w")

    summary.write("# " + field_name + " - 2017 Temperature Data")
    summary.write("\n\n")

    summary.write("***")
    summary.write("\n\n")
    summary.write("### Data Overview")
    summary.write("\n\n")

    summary.write("- Number of Measurements [#] = " + str(measurements) + '\n')

    summary.write("- Average Air Temperature [C] = "
                  + "{:.2f}".format(mean_air) + '\n')
    summary.write("- Standard Deviation for Air Temperature [C] = "
                  + "{:.2f}".format(std_air) + '\n')
    summary.write("- Average Soil Temperature [C] = "
                  + "{:.2f}".format(mean_soil) + '\n')
    summary.write("- Standard Deviation for Soil Temperature [C] = "
                  + "{:.2f}".format(std_soil) + '\n')

    summary.write("- Highest Air Temperature [C] = " + str(max_air) + '\n')
    summary.write("- Lowest Air Temperature [C] = " + str(min_air) + '\n')
    summary.write("- Highest Soil Temperature [C] = " + str(max_soil) + '\n')
    summary.write("- Lowest Soil Temperature [C] = " + str(min_soil) + '\n')

    summary.write("- Missing Air Temperature Data = " + str(missing_air)
                  + " (" + "{:.2f}".format(pm_air) + r"%)" + '\n')
    summary.write("- Missing Soil Temperature Data = " + str(missing_soil)
                  + " (" + "{:.2f}".format(pm_soil) + r"%)" + '\n')

    summary.write("\n")

    summary.write("***")
    summary.write("\n\n")
    summary.write("### Yearly Air Temperature Plot")
    summary.write("\n\n")

    summary.write("![](2017_Air_Temperature_Scatter_Plot.png)")
    summary.write("\n\n")

    summary.write("***")
    summary.write("\n\n")
    summary.write("### Yearly Soil Temperature Plot")
    summary.write("\n\n")

    summary.write("![](2017_Soil_Temperature_Scatter_Plot.png)")
    summary.write("\n\n")

    summary.write("***")
    summary.write("\n\n")
    summary.write("### Summary of Air Temperature Data")
    summary.write("\n\n")

    summary.write(air_summary.to_markdown(tablefmt = "github"))
    summary.write("\n\n")

    summary.write("***")
    summary.write("\n\n")
    summary.write("### Monthly Air Temperature Plots")
    summary.write("\n\n")

    for plot in air_plot_paths:
        summary.write("![](" + plot + ')')
        summary.write("\n\n")

    summary.write("***")
    summary.write("\n\n")
    summary.write("### Summary of Soil Temperature Data")
    summary.write("\n\n")

    summary.write(soil_summary.to_markdown(tablefmt = "github"))
    summary.write("\n\n")

    summary.write("***")
    summary.write("\n\n")
    summary.write("### Monthly Soil Temperature Plots")
    summary.write("\n\n")

    for plot in soil_plot_paths:
        summary.write("![](" + plot + ')')
        summary.write("\n\n")

    summary.close()


def main():
    global status, current, total

    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "-i", "--input_path",
        type = str,
        help = "Path to directory containing .csv files of Weather Data. The "
               "script expects one file per field. Any non-csv files will be "
               "ignored.",
        required = True)
    parser.add_argument(
        "-o", "--output_path",
        type = str,
        help = "Path to directory that will contain generated field plots and "
               "summaries.",
        required = True)

    args = parser.parse_args()

    print("\n")

    files = os.listdir(args.input_path)

    total = len(files)
    tick_msg("Found data for " + str(total) + " fields.\n")

    status = True
    loading = Thread(target = anim_process,
                     args = ("Generating field summaries and plots.",))
    loading.start()

    for file in files:

        if file.split('.')[1] == "csv":

            field_name = file.split('.')[0]

            # Open data for current field.
            file_path = args.input_path + '/' + file
            field_df = pd.read_csv(file_path)

            # Directory to store files for the current field.
            directory_path = args.output_path + '/' + field_name
            os.mkdir(directory_path)

            month_list = field_df["Month"].unique()

            plot_year(field_df, field_name, month_list,
                      "Scatter", directory_path)
            plot_year(field_df, field_name, month_list,
                      "Linear", directory_path)

            for month in month_list:
                month_df = field_df[field_df["Month"] == month]
                plot_month(month_df, field_name, month,
                           "Scatter", directory_path)
                plot_month(month_df, field_name, month,
                           "Linear", directory_path)

            field_summary(field_df, field_name, directory_path)

        current += 1

    status = False
    loading.join()
    tick_msg("Successfully generated field summaries and plots.\n")

    tick_msg("Done!")


if __name__ == "__main__":
    main()
