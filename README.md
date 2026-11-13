# ScienceQA Multimodal Question Answering

## Project Overview

This project compares three machine learning models for answering multiple-choice questions using text and images from the ScienceQA-IMG dataset.

The aim is to understand how combining text and image features affects question-answering accuracy.

## Dataset

The project uses the ScienceQA-IMG dataset.

After removing overlapping images and question text across dataset splits, the cleaned dataset contains:

- Training: 6,218 questions
- Validation: 1,080 questions
- Test: 1,038 questions

## Models

Three models were implemented:

1. **BERT Text-only:** Uses BERT text features without image information.
2. **Early Fusion:** Combines BERT text features and ViT image features.
3. **Intermediate Fusion:** Uses eight-head attention to combine BERT text features with ViT image patch features.

Pretrained BERT and ViT encoders were kept frozen during training.

## Experimental Setup

The models were trained in Google Colab using PyTorch and GPU support.

- Optimizer: Adam
- Learning rate: 0.001
- Training epochs: 20
- Loss function: Cross-entropy
- Random seeds: 42, 123 and 2026

The best checkpoints were selected using validation loss.

## Results

The following results show the mean test accuracy across three random seeds.

| Model | Mean Test Accuracy | Standard Deviation |
|---|---:|---:|
| BERT Text-only | 44.54% | 1.20% |
| Early Fusion | 48.97% | 0.40% |
| Intermediate Fusion | 82.53% | 1.43% |

Intermediate fusion achieved the highest accuracy across all three runs.

## Additional Evaluation

The project also includes:

- Image-shuffling experiments
- Subject-wise and grade-wise performance analysis
- Overfitting analysis
- Bootstrap confidence intervals
- Model calibration
- Inference time and GPU memory comparison
- Error analysis

## How to Run

1. Open `ScienceQA_Project_2.2.ipynb` in Google Colab.
2. Enable a GPU runtime.
3. Run the notebook cells in order.
4. The notebook downloads the dataset and pretrained models and generates the required features and checkpoints.

Some experiments require significant GPU memory and training time. Saved checkpoints and extracted features are generated during execution and are not included in this repository.

## Conclusion

The results show that attention-based intermediate fusion achieved better question-answering accuracy than the text-only and early-fusion models. However, it required more computation and sometimes produced incorrect answers with high confidence.
