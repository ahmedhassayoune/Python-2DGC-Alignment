#! /usr/bin/Rscript
library(RGCxGC)

args = commandArgs(trailingOnly=TRUE)
print(args)
REF_PATH = args[1]


ref = read_chrom(REF_PATH, mod_time = 1.25, verbose = F)

batch_samples <- list()
k = 1
name_vec = c()
for (i in c(2:length(args))) {

    name = gsub(" ", "", paste("Chrom", k))
    batch_samples <- append(batch_samples, read_chrom(args[i], mod_time = 1.25, verbose = F))
    name_vec <- c(name_vec, name)
    k <- k + 1
}

batch_samples <- setNames(batch_samples, name_vec)


batch_alignment <- batch_2DCOW(ref, batch_samples, c(10, 40), c(1,10))
write.table(t(ref@chromatogram), file=gsub(" ", "", paste(substring(args[1], 1, nchar(args[1]) - 3),"txt")), row.names = F, col.names = F)

j = 2

for (chrom in batch_alignment@Batch_2DCOW) {
    tmp <- substring(args[j], 1, nchar(args[j]) - 3)
    tmp <- (gsub(" ", "", paste(tmp,"txt")))
    write.table(t(chrom), file=tmp, row.names = F, col.names = F)
    j <- j + 1
}