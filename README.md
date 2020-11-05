# Introduction
I try to use pretrained chinese model, such as bert-base-chinses , XLNet-mid and XLNet-base and so on, to solve some NLP task.
 
In this task, I put the model into a document classification task, which was got in a online contest.
  
# Prerequisites
Python 3.6  
[Pytorch](https://pytorch.org/) 1.1.0+
[CUDA](https://developer.nvidia.com) GPU 2070S+ 
[bert-base-chinese](https://huggingface.co/bert-base-chinese#) - My first pretrained bert model is this.   
# Descriptions
**Data** - A dir where contains resources used in this code.  
* ```labeled_data.csv```: The dataset that is used for training with labels.  
* ```unlabeled_data.csv```:  The dataset that is used for training without labels.  
* ```test_data.csv```: The dataset that is used for testing.  
* ```model.pth```: The file that reserves the information of model and optimizer.  

```pred.csv``` - The file that reserves the prediction results.  
```result.csv``` - The file that reserves the final results..  
# Usage
bert_pre_nlp.py
