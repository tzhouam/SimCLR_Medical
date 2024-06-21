# Medical Image Classification with SimCLR

This README file provides an overview of a study on the application of SimCLR (Simple Framework for Contrastive Learning of Visual Representations) in the domain of medical image classification, specifically focusing on the Fitzpatrick17k dataset. The study aims to evaluate the performance of SimCLR when combined with traditional deep learning models and its potential to advance computer-aided diagnosis in medical imaging.

## Summary

The study introduces SimCLR as a contrastive learning framework for medical image classification. By leveraging data augmentation techniques, SimCLR creates different views of the same image to enhance the learning of abstract visual representations without requiring memory banks. The research demonstrates that integrating SimCLR with traditional deep learning models improves classification accuracy and mitigates overfitting, particularly in scenarios with limited labeled data.

## Key Findings

- **Enhanced Classification Accuracy**: The combination of SimCLR with traditional deep learning models such as ResNet and DenseNet results in improved classification accuracy in medical image classification tasks.
  
- **Mitigated Overfitting**: SimCLR shows promise in reducing overfitting, especially when working with limited labeled data, such as 1% or 10% of the entire dataset.

## Questions Addressed

- **Data Augmentation Techniques**: The study explores how SimCLR leverages data augmentation techniques, such as cropping, resizing, flipping, color distortion, and Gaussian blur, to create different views of the same image, thereby enhancing the learning of abstract visual representations.
  
- **Performance and Overfitting Comparison**: A comparison is made between SimCLR combined with traditional deep learning models and traditional supervised learning methods, highlighting the improved performance and reduced overfitting achieved with SimCLR.

## Limitations and Challenges

The study acknowledges potential limitations associated with the quality and diversity of data augmentations in medical imaging, the generalizability of findings to other types of medical images, and the initial requirement for a substantial amount of unlabeled data.
