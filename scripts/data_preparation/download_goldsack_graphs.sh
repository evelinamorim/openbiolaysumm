#!/bin/bash
#
# Download Original Goldsack et al. Graph Data
# 
# This script downloads the graph data from Goldsack et al. (2023) for
# comparison purposes. This is OPTIONAL - only needed if you want to
# compare with their original preprocessed graphs.
#
# Source: https://github.com/TGoldsack1/Enhancing_Biomedical_Lay_Summarisation_with_External_Knowledge_Graphs
#
# Usage:
#   bash scripts/data_preparation/download_goldsack_graphs.sh
#

mkdir -p data

# Graph data
fileid="14cHpqW3b6upcCvwMUX-ZC1aFP4_0Zai5"
filepath="./data/graph_data.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filepath}

unzip ./data/graph_data.zip -d ./data/