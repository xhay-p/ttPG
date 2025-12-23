# ttPG - Torch and Transformers Playground

A comprehensive learning repository containing Jupyter notebooks and Python scripts for mastering PyTorch and Hugging Face Transformers. This repository serves as a hands-on learning resource covering fundamental concepts to advanced techniques in deep learning and natural language processing.

## 📚 Repository Structure

### 🔥 PyTorch Basics (`pytorch_basics/`)
Fundamental PyTorch concepts and operations:
- **`tensors.ipynb`** - Introduction to PyTorch tensors, data types, and basic operations
- **`derivatives.ipynb`** - Understanding automatic differentiation and gradients
- **`dataset-dataloader.ipynb`** - Working with datasets and data loaders for efficient data handling

### 🤗 Transformers Basics (`transformers_basics/`)
Hugging Face Transformers library fundamentals:
- **`pipeline_module.ipynb`** - Using pre-built pipelines for common NLP tasks
- **`tokenizer.ipynb`** - Text tokenization and preprocessing
- **`models.ipynb`** - Working with pre-trained models
- **`datasets.ipynb`** - Loading and processing datasets
- **`trainer_finetuning.ipynb`** - Fine-tuning models using the Trainer API
- **`behind_pipeline.ipynb`** - Understanding the internals of transformer pipelines

### 🎯 Sentiment Classification (`sentiment_classification/`)
Complete sentiment analysis project:
- **`sample_notebook/notebook.ipynb`** - Sample implementation and experimentation
- **`scripts/train.py`** - Training script for sentiment classification using DistilBERT
- **`scripts/eval.py`** - Evaluation script for model performance assessment

### ⚡ Model Quantization (`model_quantization/`)
Advanced techniques for model optimization:
- **`llm_quantizaton_inference.ipynb`** - Large Language Model quantization techniques for efficient inference

### 📊 Text Feature Exploration
- **`text_feature_exploration.ipynb`** - Text feature extraction and visualization techniques
- **`word_embeddings_gensim_word2vec.ipynb`** - Word2Vec implementation using Gensim

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for some notebooks)

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/ttPG.git
cd ttPG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies
- `torch` - PyTorch deep learning framework
- `torchtext` - Text processing utilities
- `transformers` - Hugging Face Transformers library
- `accelerate` - Training acceleration utilities
- `bitsandbytes` - Quantization and optimization
- `einops` - Tensor operations
- `sentence_transformers` - Sentence embeddings



## 🤝 References

### Official Learning Resources
- **[PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)** - Official PyTorch tutorial covering tensors, datasets, neural networks, and complete ML workflows
- **[Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1)** - Comprehensive course on Large Language Models and NLP using the Hugging Face ecosystem

### Additional Resources
- [PyTorch Documentation](https://pytorch.org/docs/) - Complete PyTorch API reference
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/) - Official Transformers library documentation
- [Hugging Face Hub](https://huggingface.co/) - Discover and share models, datasets, and demos


## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Hugging Face for the comprehensive transformers library
- The open-source community for continuous inspiration and support
