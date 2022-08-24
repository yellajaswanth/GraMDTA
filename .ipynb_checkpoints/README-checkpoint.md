# GraMDTA: Multimodal Graph Neural Networks for Predicting Drug-Target Associations

Authors: Jaswanth K. Yella, Sudhir K. Ghandikota, Anil G. Jegga

## Abstract

Finding novel drug-target associations is vital for drug discovery. However, screening millions of small molecules for a select target protein is challenging. Several computational approaches have been proposed in the past using machine learning methods to find the candidate drugs for proteins. Some of these works utilized structures of drugs and proteins for modeling. A few of the works utilized knowledge graph networks and identified the potential candidates through link prediction approaches. While structural learning offers molecular-based representations, the knowledge graph-based learning offers interaction-based representations. Such multimodal sources of information acting complimentarily could improve the robustness of drug-target association (DTA) predictions. In this work, we propose multimodal graph neural network to learn both structural and knowledge graph representations while utilizing multi-head attention to fuse the multimodal representations and predict DTAs. We compare our proposed approach with existing works and show the benefits of multimodal fusion for DTA.

![image](https://freeimage.host/i/r85ZPe)



## Dependencies
* PyTorch
* PyTorch Sparse
* PyTorch Geometric
* Numpy
* Pandas
* Scikit-learn
* Networkx