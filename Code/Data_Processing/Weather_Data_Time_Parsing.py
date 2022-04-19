"""
Weather Data Time Parsing

This script parses the time strings contained in the G2F Weather Data and
separates them into respective columns for Hours, Minutes, Seconds, and Period.

Additionally, it also calculates the equivalent Decimal Time (in hours) for each
measurement.

The Weather Data will be saved in separate files corresponding to each field.

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


def parse_time(time_str):
    """
    Parses a given time string into separate strings for the Hours, Minutes,
    Seconds, and Period.

    Parameters
    ----------
    time_str : str
        Time string to be parsed in the form of "HH:MM:SS P".

    Returns
    -------
    h : str
        Number of hours.
    m : str
        Number of minutes.
    s : str
        Number of seconds.
    p : str
        Period, in the form of "AM" or "PM".
    """
    h, m, s = time_str.split(':')
    s, p = s.split(' ')

    return h, m, s, p


def decimal_time(h, m, s, p):
    """
    Converts a given time measurement into decimal time, in hours.

    Parameters
    ----------
    h : str
        Number of hours.
    m : str
        Number of minutes.
    s : str
        Number of seconds.
    p : str
        Period, in the form of "AM" or "PM".

    Returns
    -------
    float
        Equivalent decimal time, in hours.
    """
    if p == "PM" and h != 12:
        h = h + 12

    if p == "AM" and h == 12:
        h = 0

    return h + (m / 60) + (s / 3600)


def main():
    global status, current, total

    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "-w", "--weather",
        type = str,
        help = "Path to .csv file containing Weather Data.",
        required = True)
    parser.add_argument(
        "-d", "--directory",
        type = str,
        help = "Path to directory that will contain the output files for each "
               "field present in given data.",
        required = True)

    args = parser.parse_args()

    print("\n")

    status = True
    loading = Thread(target = anim_loading,
                     args = ("Loading Weather Data into a Pandas DataFrame.",))
    loading.start()

    weather_df = pd.read_csv(args.weather)

    status = False
    loading.join()
    tick_msg("Successfully loaded Weather Data into a Pandas DataFrame.")

    info_msg("Total Weather Measurements (#) = " + str(weather_df.shape[0]))

    field_list = weather_df["Field Location"].unique()

    info_msg("Found data for " + str(field_list.shape[0]) + " fields.\n")

    for field in field_list:

        info_msg("Parsing " + field + " field.")

        field_df = weather_df[weather_df["Field Location"] == field].copy()
        field_df.reset_index(drop = True, inplace = True)

        info_msg(field + " Weather Measurements (#) = " +
                 str(field_df.shape[0]))

        month_list = field_df["Month"].unique()

        current = 0
        status = True
        loading = Thread(target = anim_process,
                         args = ("Parsing monthly data for " + field + '.',))
        loading.start()

        total = month_list.shape[0]

        for month in month_list:

            month_df = field_df[field_df["Month"] == month]
            day_list = month_df["Day"].unique()

            for day in day_list:

                day_df = month_df[month_df["Day"] == day]

                for i in day_df.index.values:

                    h, m, s, p = parse_time(field_df.loc[i, "Time"])

                    field_df.loc[i, "Hours"] = int(h)
                    field_df.loc[i, "Minutes"] = int(m)
                    field_df.loc[i, "Seconds"] = int(s)
                    field_df.loc[i, "Period"] = p

            month_df = field_df[field_df["Month"] == month]
            day_list = month_df["Day"].unique()

            for day in day_list:

                day_df = month_df[month_df["Day"] == day]
                hours = day_df["Hours"].to_numpy()
                indexes = day_df.index.to_numpy()
                max_idx = 0

                for i in range(0, indexes.size - 1):
                    if (hours[i] == 11) and (hours[i + 1] == 12):
                        max_idx = indexes[i]

                if max_idx == 0:
                    max_idx = np.max(day_df.index.values)

                field_df.loc[
                    (field_df.index <= max_idx) &
                    (field_df.index >= np.min(day_df.index.values)),
                    "Period"] = "AM"

                field_df.loc[
                    (field_df.index > max_idx) &
                    (field_df.index <= np.max(day_df.index.values)),
                    "Period"] = "PM"

                for i in day_df.index.values:

                    h = field_df.loc[i, "Hours"]
                    m = field_df.loc[i, "Minutes"]
                    s = field_df.loc[i, "Seconds"]
                    p = field_df.loc[i, "Period"]

                    field_df.loc[i, "Decimal Time"] = decimal_time(h, m, s, p)

            current += 1

        status = False
        loading.join()
        tick_msg("Successfully parsed monthly data for " + field + '.')

        status = True
        loading = Thread(target = anim_loading,
                         args = ("Saving modified " + field + "DataFrame to "
                                                              ".csv file.",))
        loading.start()

        field_df.to_csv(args.directory + '/' + field + ".csv", index = False)

        status = False
        loading.join()
        tick_msg("Successfully saved modified " + field + " DataFrame to "
                                                          ".csv file.\n")

    tick_msg("Done!")


if __name__ == "__main__":
    main()
