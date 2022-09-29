
# K-LM: Knowledge Agumenting in Langauge-Models 

```
This repository contains the source code of the project.  
```
### Requirements
Software:
```
Python3
Pytorch >= 1.0
argparse == 1.1
```
### Knowledge Graphs
```
The Artificial Intelligence Knowledge Graph (AI-KG) dump can be downloded in .ttl format from the link below:
```
URL: https://scholkg.kmi.open.ac.uk/

### Publication (IEEE Access)
### 1. K-LM: Knowledge Augmenting in Language Models Within the Scholarly Domain
URL: https://ieeexplore.ieee.org/document/9866735 
```
Abstract: The use of superior algorithms and complex architectures in language models have successfully imparted human-like abilities to machines for specific tasks. But two significant constraints, the available training data size and the understanding of domain-specific context, hamper the pre-trained language models from optimal and reliable performance. A potential solution to tackle these limitations is to equip the language models with domain knowledge. While the commonly adopted techniques use Knowledge Graphs Embeddings (KGEs) to inject domain knowledge, we provide a Knowledge Language Model (K-LM) to use the Resource Description Framework (RDF) triples directly, extracted from world knowledge bases. The proposed model works in conjunction with Generative Pretrained Transformer (GPT-2) and Bidirectional Encoder Representations from Transformers (BERT) and uses a well-defined pipeline to select, categorize, and filter the RDF triples. In addition, we introduce heuristic methods to inject domain-specific knowledge in K-LM, leveraging knowledge graphs (KGs). We tested our approaches on the classification task within the scholarly domain using two KGs, and our results show that our proposed language model has significantly outperformed the baselines and BERT for each KG. Our experimental findings also help us conclude the importance of relevance of KG used over the quantity of injected RDF triples. Also, each of our proposed methods for injecting the RDF triples has increased the overall model’s accuracy, demonstrating that K-LM is a potential choice for domain adaptation to solve knowledge-driven problems.
```
