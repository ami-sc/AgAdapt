"""
CSV To H5 File Conversion

This script extracts data from a .csv file containing Genotype Data and writes
it into a .h5 file.

**** AgAdapt Project ****
"""


import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "-csv", "--csv_file",
        type = str,
        help = "Path to .csv file to read data from.",
        required = True)
    parser.add_argument(
        "-h5", "--h5_file",
        type = str,
        help = "Path to .h5 file to write data to.",
        required = True)
    parser.add_argument(
        "-h5k", "--h5_key",
        type = str,
        help = "File key for the .h5 file.",
        required = True)

    args = parser.parse_args()

    print("\n")

    print("[ > ] Extracting data from .csv file.")
    genotype_df = pd.read_csv(args.csv_file, index_col = 0)

    print("[ > ] Saving data into .h5 file.\n")
    genotype_df.to_hdf(args.h5_file, args.h5_key)

    print("[ > ] Done!")


if __name__ == "__main__":
    main()
