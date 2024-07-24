# KGE_literature_relevance_assessment
This repository focuses on the RELISH Corpus to identify relevancy of a given pair of PubMed papers. The approach uses various knowledge graph embedding (KGE) models such as TransE, RotatE and ConvE, to train entitiy embeddings which are used to computes a semantic closeness between two documents. The script includes the knowledge graph construction pipeline, which uses OpenAlex API to retrieve knowledge bases and then construct knowledge graph triples that are being used as training input for the KGE models.

## Table of Contents

1. [About](#about)
2. [Input Data](#input-data)
4. [Code Implementation](#code-implementation)
5. [Getting Started](#getting-started)

## About

This is the complete pipeline from retrieval of knowledge graph triples to the training of knowledge graph embedding (KGE), as well as ranking-based evaluation strategies of the RElevant LIterature SearcH (RELISH) corpus.

## Input Data

 The input of this approach is the 3 column Relevance Matrix generated from the original RELISH dataset using the script of the [OntoClue preprocessing pipeline](https://github.com/zbmed-semtec/relish-preprocessing).

## Getting Started

To get started with this project, follow these steps:

### Step 1: Clone the Repository
First, clone the repository to your local machine using the following command:

###### Using HTTP:

```
git clone https://github.com/zbmed-semtec/KGE_literature_relevance_assessment.git
```

###### Using SSH:
Ensure you have set up SSH keys in your GitHub account.

```
git clone git@github.com:zbmed-semtec/KGE_literature_relevance_assessment.git
```
### Step 2: Create a virtual environment and install dependencies

To create a virtual environment within your repository, run the following command:

```
python3 -m venv .venv 
source .venv/bin/activate   # On Windows, use '.venv\Scripts\activate' 
```

To confirm if the virtual environment is activated and check the location of yourPython interpreter, run the following command:

```
which python    # On Windows command prompt, use 'where python'
                # On Windows PowerShell, use 'Get-Command python'
```
The code is stable with python 3.6 and higher. The required python packages are listed in the requirements.txt file. To install the required packages, run the following command:

```
pip install -r requirements.txt
```

To deactivate the virtual environment after running the project, run the following command:

```
deactivate
```

### Step 3: Retrieve OpenAlex knowledge base
The ['Knowledge_Base_Retrieval.py'](./code/knowledge_base/Knowledge_Base_Retrieval.py) script uses the 3 column RELISH relevance matrix ['RELISH_Relevance_Matrix.tsv'](./data//RELISH_Relevance_Matrix.tsv) as input and performs a series of API calls to retrieve the OpenAlex knowledge base for each unique PMID and saves it as a JSON.

```
python3 code/knowledge_base/Knowledge_Base_Retrieval.py [-i RELISH RELEVANCE MATRIX PATH] [-o OUTPUT PATH OPENALEX KNOWLEDGE BASE]
```

You must pass the following two arguments:

+ -i/ --input: Path for RELISH 3 column TSV file.
+ -o/ --output: Path for OpenAlex knowledge base JSON.

To run this script, please execute the following command:

```
python3 code/knowledge_base/Knowledge_Base_Retrieval.py -i data/RELISH_Relevance_Matrix.tsv -o data/OpenAlex_KB.json
```

### Step 4: Retrieve OpenAlex citation
The ['Citation_Retrieval.py'](./code/knowledge_base/Citation_Retrieval.py) script uses a series of API calls using the previously generated OpenAlex knowledge base
to retrieve all "cited by" attributes for each PMID in the knowledge base JSON. The resulting cited by information gets saved as a JSON.

```
python3 code/knowledge_base/Citation_Retrieval.py [-i OPENALEX KNOWLEDGE BASE JSON PATH] [-o OUTPUT PATH OPENALEX CITED BY JSON]
```

You must pass the following two arguments:

+ -i/ --input: Path for input OpenAlex knowledge base JSON.
+ -o/ --output: Path for cited-by JSON.

To run this script, please execute the following command:

```
code/knowledge_base/Citation_Retrieval.py -i data/OpenAlex_KB.json -o data/cited_by.json
```

### Step 5: Triples construction
The ['Triples_Construction.py'](./code/Triples_Construction.py) uses both the OpenAlex KG and the cited by JSON to generate a three column triples TSV file,
representing a directed knowledge graph to be used to train the KGE models.

```
python3 code/Triples_Construction.py [-ik OPENALEX KNOWLEDGE BASE JSON] PATH [-ic OPENALEX CITED BY JSON PATH] [-ir RELISH RELEVANCE MATRIX TSV] [-o OUTPUT TRIPLES DIRECTORY PATH]
```

You must pass the following four arguments:

+ -ik/ --inputKG: Path for input OpenAlex knowledge base JSON.
+ -ic/ --inputCitation: Path for input cited-by JSON.
+ -ir/ --inputRelevanceMatrix: Input of RELISH PMID relevance matrix TSV.
+ -o/ --output: Directory path for chunked triples.

To run this script, please execute the following command:

```
python3 code/Triples_Construction.py -ik data/OpenAlex_KB.json -ic data/cited_by.json -ir data/RELISH_Relevance_Matrix.tsv -o data/Chunked_Triples
```

### Step 6: Generate Embeddings and Calculate Cosine Similarity
The [`Train_KGE.py`](./code/Train_KGE.py) script trains KGE models and returns NPY files of entity embeddings and all entity labels.
The parameter options allow for switching of the training approach between stochastic Local Closed World Assumption (sLCWA) and Local Closed World Assumption (LCWA),
as well we setting the KGE model, for which these models can be picked: TransE, TransH, TransR, TransD, RotatE and ConvE.

```
python3 code/Train_KGE.py [-ik OPENALEX KNOWLEDGE BASE JSON PATH] [-it TRIPLES TSV PATH] [-ir RELISH RELEVANCE MATRIX PATH] [-m KGE MODEL NAME] [-t TRAINING APPROACH NAME] [-g USE GPU] [-oe OUTPUT EMBEDDINGS DIRECTORY] [-ot OUTPUT TRIPLES DIRECTORY] [-op OUTPUT PROCESSING TIME JSON] [-oc OUTPUT COSINE SIMILARITY MATRIX TSV]
```

You must pass the following nine arguments:

+ -ik/ --inputKG: Path for input OpenAlex knowledge base JSON.
+ -it/ --inputTriples: Directory path for input triples.
+ -ir/ --inputRelevanceMatrix: Input of RELISH PMID relevance matrix TSV.
+ -m/ --kgeModel: Name of knowledge graph embedding model.
+ -t/ --trainingApproach: Name of training approach, it can either be sLCWA or LCWA.
+ -g/ --useGPU: 1 if CUDA GPU should be used for training and 0 if the CPU should be used instead
+ -oe/ --outputEmbeddings: Directory path for embeddings output.
+ -ot/ --outputTriples: Directory path for triples output.
+ -op/ --outputProcessingTime: Output of processing time JSON.
+ -oc/ --outputCosineSimilarity: Output of cosine similarity matrix TSV.

To run this script, please execute the following command:

```
python3 code/Train_KGE.py -ik data/OpenAlex_KB.json -it data/Chunked_Triples -ir data/RELISH_Relevance_Matrix.tsv -m TransE -t sLCWA -oe data/embeddings -g 1 -ot data/triples -op data/processing_time.json -oc data/RELISH_Cosine_Similarity.tsv
```

### Step 7: Precision@N
In order to calculate the Precision@N scores and execute this [script](/code/evaluation/precision.py), run the following command:

```
python3 code/evaluation/precision.py [-i COSINE SIMILIRATY TSV PATH] [-o OUTPUT PRECISION TSV]
```

You must pass the following three arguments:

+ -i/ --wmd_file_path: File path to the 4-column cosine similarity RELISH TSV file.
+ -o/ --output_path: path to save the generated precision matrix: (tsv file).
+ -m/ --multiple_classes: If 1, apply the 3-class approach, if 0 apply the 2-class approach of considering partially-relevant articles to be positive.

To run this script, please execute the following command:

```
python3 code/evaluation/precision.py -i data/RELISH_Cosine_Similarity.tsv -o data/precision_KGE_3c.tsv -m 1
```


### Step 8: nDCG@N
In order to calculate nDCG scores and execute this [script](/code/evaluation/calculate_gain.py), run the following command:

```
python3 code/evaluation/calculate_gain.py [-i COSINE SIMILIRATY TSV PATH]  [-o OUTPUT]
```

You must pass the following two arguments:

+ -i / --input: Path to the 4 column cosine similarity matrix RELISH TSV file.
+ -o/ --output: Output path along with the name of the file to save the generated nDCG@N TSV file.

To run this script, please execute the following command:

```
python3 code/evaluation/calculate_gain.py -i data/RELISH_Cosine_Similarity.tsv -o data/nCDG_KGE.tsv
```