# GraMDTA: Multimodal Graph Neural Networks for Predicting Drug-Target Associations
  
Authors: Jaswanth K. Yella, Sudhir K. Ghandikota, Anil G. Jegga

## Abstract

Finding novel drug-target associations is vital for drug discovery. However, screening millions of small molecules for a select target protein is challenging. Several computational approaches have been proposed in the past using machine learning methods to find the candidate drugs for proteins. Some of these works utilized structures of drugs and proteins for modeling. A few of the works utilized knowledge graph networks and identified the potential candidates through link prediction approaches. While structural learning offers molecular-based representations, the knowledge graph-based learning offers interaction-based representations. Such multimodal sources of information acting complimentarily could improve the robustness of drug-target association (DTA) predictions. In this work, we propose multimodal graph neural network to learn both structural and knowledge graph representations while utilizing multi-head attention to fuse the multimodal representations and predict DTAs. We compare our proposed approach with existing works and show the benefits of multimodal fusion for DTA.

![image](https://i.ibb.co/VBH6z0C/Modal-Rx-Architecture.png)


## Dependencies
```
conda create --name GraMDTA python==3.8.12
conda activate GraMDTA
conda install pandas==1.4.1 matplotlib scikit-learn==1.0.2 scipy==1.8.1 tqdm seaborn==0.11.2 tensorboard==2.8.0
conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch
conda install rdkit -c conda-forge

pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip3 install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip3 install torch-geometric

pip3 install -U pytorch_warmup
pip3 install -U networkx==2.8.4

```
