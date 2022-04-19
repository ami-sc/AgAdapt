`/Data_Processing`: This directory contains 7 files:

- `Genotype_Data_Cleaning.py`:
  - This script filters and discards individuals with a target amount of missing data.

- `Genotype_Data_Filtering.py`:
  - This script filters Single Nucleotide Polymorphism (SNP) data.
  - Filtering is performed according to a provided metric and a specified threshold.

- `Genotype_Data_Imputation.py`:
  - This script imputes missing Genotype Data, per chromosome position.
  - Missing data (NaN values) are replaced with the most frequently observed allele in a given position.

- `Genotype_Data_Metrics.py`:
  - This script calculates general metrics for the Genotype Data of a set of individuals.
  - Metrics are calculated per chromosome position.
  - Intended to be used as a basis for data filtering.
  - Calculated metrics include:
    - Total Individuals
    - Number and Percentage of Homozygous Mutant Individuals
    - Number and Percentage of Heterozygous Individuals
    - Number and Percentage of Homozygous WildType Individuals
    - Observed Copies and Frequency of Mutant Allele
    - Observed Copies and Frequency of WildType Allele

- `Master_Dataset_Generation.py`:
  - This script concatenates Genotype, Phenotype, and Environmental Features into single, per-field Master Datasets.

- `Offspring_Genotype_Prediction.py`:
  - This script predicts Genotype Data for target offspring individuals, given the Genotype Data of both parents and the corresponding pedigree of the offspring.
  - Prediction is done by incorporating parental data with average alleles per position.

- `Weather_Data_Time_Parsing.py`:
  - This script parses the time strings contained in the G2F Weather Data file and separates them into respective columns for Hours, Minutes, Seconds, and Period. 
  - Additionally, it also calculates the equivalent Decimal Time (in hours) for each measurement. 
  - Paring is done per field.
  - Separate `.csv` files will be generated for each field.
