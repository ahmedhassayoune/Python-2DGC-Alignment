#! /usr/bin/Rscript
library(R2DGC)
PATH = commandArgs(trailingOnly=TRUE)
sampleList = list.files(path=PATH, full.names = T,pattern =".txt")
print(sampleList)
sampleListNoProcessed = c()
for (sample in sampleList) {
    if (grepl("Processed", sample) == FALSE)
        sampleListNoProcessed = c(sampleListNoProcessed, sample)
}
sampleList = sampleListNoProcessed
for (sample in sampleList) {
    PrecompressFiles(inputFileList=sample, outputFiles=T, RT1Penalty=1, RT2Penalty=0.01)
}

processedSampleList = list.files(path=PATH, full.names = T, pattern ='*_Processed.txt')
Alignment<-ConsensusAlign(processedSampleList, missingValueLimit=0.5)
colnames(Alignment$Alignment_Matrix)<-gsub("^.+/","",colnames(Alignment$Alignment_Matrix))
write.csv(Alignment$Alignment_Matrix, file.path(PATH,"aligned_peak_table.csv"), row.names=T)
