#!/bin/bash

echo "Downloading real sequencing data from EMBL-ENA..."
echo "This will download ~1-2 GB of data and may take 10-30 minutes"
echo ""

# Create directory for real data
mkdir -p real_data
cd real_data

# Download ERR240727_1 (100bp reads)
echo "Downloading ERR240727_1 (100bp reads)..."
wget -c ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR240/ERR240727/ERR240727_1.fastq.gz

# Download SRR826460_1 (150bp reads) 
echo "Downloading SRR826460_1 (150bp reads)..."
wget -c ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR826/SRR826460/SRR826460_1.fastq.gz

# Download SRR826471_1 (250bp reads)
echo "Downloading SRR826471_1 (250bp reads)..."
wget -c ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR826/SRR826471/SRR826471_1.fastq.gz

echo ""
echo "Uncompressing files..."
gunzip -k *.fastq.gz

echo ""
echo "Download complete!"
ls -lh *.fastq
