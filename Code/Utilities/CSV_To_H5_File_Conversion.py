"""CSV To H5 File Conversion

This script extracts data from a .csv file containing Genotype Data and writes
it into a .h5 file.

Required Libraries:
    - argparse
    - pandas

**** AgAdapt Project ****
"""


import argparse
import pandas as pd


def csv_to_h5(csv_file, h5_file, h5_key):
    """Converts a .csv file to a .h5 file.

    Parameters
    ----------
    csv_file : str
        Path to .csv file to read data from.
    h5_file  : str
        Path to .h5 file to write data to.
    h5_key   : str
        File key for the .h5 file.
    """

    print("\n\n[ > ] Extracting data from .csv file.")
    genotype_df = pd.read_csv(csv_file, index_col = 0)

    print("[ > ] Saving data into .h5 file.")
    genotype_df.to_hdf(h5_file, h5_key)

    print("[ > ] Done!")


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "--csv_file",
        type = str,
        help = "Path to .csv file to read data from.")

    parser.add_argument(
        "--h5_file",
        type = str,
        help = "Path to .h5 file to write data to.")

    parser.add_argument(
        "--h5_key",
        type = str,
        help = "File key for the .h5 file.",)

    args = parser.parse_args()
    csv_to_h5(args.csv_file, args.h5_file, args.h5_key)


if __name__ == "__main__":
    main()
