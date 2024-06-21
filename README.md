# Medical Image Classification with SimCLR

This project provides a glance at the application of SimCLR (Simple Framework for Contrastive Learning of Visual Representations) in medical image classification, specifically focusing on the Fitzpatrick17k dataset. The study aims to evaluate the performance of SimCLR when combined with traditional deep learning models and its potential to advance computer-aided diagnosis in medical imaging.

## Summary

We introduce SimCLR as a contrastive learning framework for the medical image classification field. By leveraging data augmentation techniques, SimCLR creates different views of the same image to enhance the learning of abstract visual representations without requiring memory banks. This project demonstrates that integrating SimCLR with traditional deep learning models improves classification accuracy and mitigates overfitting in medical image classification, particularly in conditions with limited labeled data.

## Key Findings

- **Enhanced Classification Accuracy**: Combining SimCLR with traditional deep learning models such as ResNet and DenseNet improves classification accuracy in medical image classification tasks.
  
- **Mitigated Overfitting**: SimCLR shows promise in reducing overfitting, especially when working with limited labeled data, such as 1% or 10% of the entire dataset.
and reduced overfitting achieved with SimCLR.

## Limitations and Challenges

We acknowledge potential limitations associated with the quality and diversity of data augmentations in medical imaging, the generalizability of findings to other medical images like X-Ray, MRI, etc., and the initial requirement for a substantial amount of unlabeled data.
