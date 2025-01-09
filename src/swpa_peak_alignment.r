#! /usr/bin/Rscript
source("swpa2gc.r")

# This script is used to align chromatograms using the SWPA algorithm.
#
# Args
# ----
# 1. output_dir: Path to the output directory where the csv results will be saved.
# 2. ref_path: Path to the reference chromatogram file.
# 3->n. target_path: Path to the target chromatogram file(s).
#
# Returns
# -------
# A CSV file representing the matched peaks between the reference and target chromatograms.

args = commandArgs(trailingOnly=TRUE)

# Extract reference path and target paths from args
output_dir = args[1]
ref_path = args[2]
targets_paths = args[3:length(args)] # Handle multiple target paths

print("Received Arguments:")
print(sprintf("   - Output directory: %s", output_dir))
print(sprintf("   - Reference: %s", ref_path))
print(paste("   - Targets: ", targets_paths))

# Load the reference chromatogram
ref_seq <- read.csv(ref_path)

# Iterate over each target path, apply SWPA, and save results
for (target_path in targets_paths) {
  
  print("Target Path:")
  print(target_path)
  
  # Load the target chromatogram
  target_seq <- read.csv(target_path)
  
  swd <- list(ref_seq, target_seq)
  
  # Apply SWPA algorithm
  results <- swpa2gc(
    swdata = swd,
    id1 = 2,        # Index for target
    id2 = 1,         # Index for reference
    opt = T
  )
  
  # Generate output filename by removing file extension from target path
  basename <- basename(target_path)
  basename <- sub("\\.[^.]*$", "", basename)
  
  # Save results to a CSV file
  write.csv(results$align, file=paste0(output_dir, "/", basename, ".csv") , row.names=TRUE)
}