#' VCF To CSV File Conversion
#'
#' Simple script to convert a .vcf file containing Genotype Data into a .csv
#' file.
#'
#' **** AgAdapt Project ****

library(adegenet)
library(vcfR)
library(roxygen2)

#' .vcf To .csv File Conversion
#'
#' Extracts Genotype Data from a .vcf file and writes it to a .csv file.
#'
#' @param   vcf_file   String - Path of .vcf file to extract data from.
#' @param   csv_file   String - Path of .csv file to write data to.
vcf_to_csv <- function(vcf_file, csv_file)
{
  cat("\n\n[ > ] Extracting genotype data from .vcf file.\n\n")
  genotype_data <- read.vcfR(vcf_file)

  cat("\n\n[ > ] Loading genotype data into a genlight object.\n\n")
  genotype_data <- vcfR2genlight(genotype_data)

  cat("\n\n[ > ] Summary of genotype data.\n\n")
  print(head(genotype_data))

  cat("\n\n[ > ] Converting genlight object into matrix.\n\n")
  genotype_data <- as.matrix(genotype_data)

  cat("\n\n[ > ] Saving matrix to .csv file.\n\n")
  write.table(genotype_data, file = csv_file, col.names = NA, sep = ",")

  cat("\n\n[ > ] Done!\n\n")
}
