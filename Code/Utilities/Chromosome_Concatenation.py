"""
Chromosome Concatenation

Simple script to concatenate multiple .h5 files containing Genotype Data into a
single .h5 file.

**** AgAdapt Project ****
"""


import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "-fl", "--file_list",
        type = str,
        help = "Path to .txt file containing file paths of .h5 files to be"
               "concatenated. The format of the list should be: "
               "/path/to/file file_key",
        required = True)
    parser.add_argument(
        "-o", "--out",
        type = str,
        help = "Path to .h5 file that will contain concatenated data.",
        required = True)
    parser.add_argument(
        "-ok", "--out_key",
        type = str,
        help = "File key for .h5 file that will contain concatenated data.",
        required = True)

    args = parser.parse_args()

    print("\n")

    concat_df = pd.DataFrame()

    print("[ > ] Extracting data from .txt file.\n")
    file_list = open(args.file_list, 'r')
    files = file_list.readlines()
    file_list.close()

    for file in files:
        file = file.strip()
        print("[ > ] Concatenating: " + file.split(' ')[0] + '')
        genotype_df = pd.read_hdf(file.split(' ')[0], file.split(' ')[1])
        concat_df = pd.concat([concat_df, genotype_df], axis = 1)

    print("\n[ > ] Saving concatenated data into .h5 file.\n")
    concat_df.to_hdf(args.out, args.out_key)

    print("[ > ] Done!")


if __name__ == "__main__":
    main()
