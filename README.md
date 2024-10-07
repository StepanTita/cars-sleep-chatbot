# WebAI Teak Home -- LLM

## Structure of the repo:
- `data/`: Contains raw and preprocessed data, augmented data, RAG data, and evaluation CSVs for human analysis.
- `notebooks/`: Jupyter notebooks for data analysis, model building, and evaluation.
- `models/`: Pickled logistic regression model for routing and adapters for LLaMA 3.2 1B for both `cars` and `sleep` datasets.
- `Cars-Sleep-App.py`: Streamlit web app serving as the front-end for the system.

## Project Overview:
This project implements a question-answering system for two domains: cars and sleep science. It uses a combination of data augmentation, task classification, fine-tuned language models, and Retrieval Augmented Generation (RAG) to provide accurate responses.

## Detailed Workflow:

### 1. Data Augmentation
- Initial dataset: 26 and 27 records per task (cars and sleep, respectively).
- Augmentation using LLaMA 3.1 70B to generate 100 additional records per task.
- Script: `notebooks/new-data.ipynb`

**Note:** The original datasets are reserved for testing purposes only.

### 2. Task Classification (Routing Strategy)
- Implemented TF-IDF with Logistic Regression for binary classification.
- Performance: 96% accuracy on the test set.
- Scripts: 
  - `notebooks/tf-idf.ipynb`
  - `notebooks/data.ipynb`

**Rationale:** Given the distinct domains, a simple word-counting approach proved highly effective. While further optimizations are possible (e.g., gradient boosting, transformers), the current lightweight solution provides excellent performance.

### 3. Fine-tuning LLaMA 3.2 1B
- Chosen for its efficiency and performance as an expert model.
- Fine-tuning script: `notebooks/fine-tune-llm.ipynb`
- Utilizes unsloth for easy model swapping (e.g., to LLaMA 8B or Gemma 9B) if needed.

### 4. Retrieval Augmented Generation (RAG)
- Parsed and preprocessed two open-source books on sleep science and car history.
- Scripts:
  - `notebooks/clean-rag-text.ipynb`: Text cleaning for RAG
  - `notebooks/langchain.ipynb`: Integrating RAG into the LangChain pipeline and running predictions

### 5. Evaluation
- Script: `notebooks/evaluate.ipynb`
- Uses the original, untouched datasets for testing.
- Employs LLaMA 3.1 70B as an automated judge for response evaluation.

### 6. Inference Pipeline
- Implemented using Streamlit for a user-friendly interface.
- To run locally:
  ```bash
  streamlit run Cars-Sleep-App.py
  ```
  **Note:** Not compatible with Mac due to BitsAndBytes lacking MPS support.

## Completed Tasks:
- [x] Use LLaMA 3.2 1B as expert model
- [x] Implement TF-IDF with logistic regression for task classification
- [x] Incorporate citation information for RAG
- [x] Augment data with responses from LLaMA 3.1 70B
- [x] Use LLaMA 3.1 70B as the judge for response evaluation
- [x] Evaluate and compare performance with and without RAG

## Requirements:
- Python 3.10+
- Streamlit
- PyTorch
- Transformers
- LangChain
- unsloth
- scikit-learn

For a complete list of dependencies, please refer to the `requirements.txt` file.



# Project Report: Question-Answering System for Cars and Sleep Science

## 1. Brief Report

### Approach:
The project implements a question-answering system for cars and sleep science domains using a combination of techniques:

1. **Data Augmentation**: Used LLaMA 3.1 70B to generate additional training data, addressing the limited initial dataset.
2. **Task Classification**: Implemented TF-IDF with Logistic Regression for efficient binary classification.
3. **Fine-tuning**: Used LLaMA 3.2 1B as the expert model, fine-tuned for each domain.
4. **RAG**: Incorporated Retrieval Augmented Generation using preprocessed books on sleep science and car history.
5. **Evaluation**: Utilized LLaMA 3.1 70B as an automated judge for response evaluation.

### Challenges faced:
1. Limited initial dataset (26-27 records per task)
2. Balancing model performance with computational efficiency
3. Ensuring clear separation between training and testing data
4. Implementing an effective routing strategy for two distinct domains
5. BitsAndBytes does not support Apple Silicon!!! (had to test through ssh)

### Reasoning behind choices:
1. **Data Augmentation**: Chose LLaMA 3.1 70B to generate diverse, high-quality additional data.
2. **Task Classification**: Opted for TF-IDF with Logistic Regression due to its simplicity and effectiveness for distinct domains (lightweight, fast, and performant).
3. **Expert Model**: Selected LLaMA 3.2 1B for its balance of efficiency and performance, with the flexibility to swap models if needed (fine-tuning and deployment pipeline is model-agnostic).
4. **RAG**: Incorporated domain-specific books to try to enhance response quality.
5. **Evaluation**: Used LLaMA 3.1 70B as an automated judge to ensure consistent and unbiased evaluation.

### Potential improvements:
1. Experiment with larger language models for better performance
2. Explore advanced RAG techniques (e.g., hybrid search, multi-document reasoning)
3. Develop a more comprehensive evaluation framework, including human evaluation
4. Optimize the Streamlit app for better performance and user experience
5. Currently conversation history is not common for 2 models in streamlit app. Sleep and cars models have separate conversation history. This needs to be fixed

## 2. Three Different Timelines

### Small Effort (Current Deliverable):
- **Components**: Data augmentation, basic task classification, fine-tuned LLaMA 3.2 1B, simple RAG, automated evaluation
- **Timeline**: ~3 days
- **Improvements**: Basic functional system with good performance for two domains

### Medium Effort:
- **Components**: All from small effort, plus:
  - Improved data augmentation with multiple LLMs
  - Advanced task classification (e.g., gradient boosting, transformers)
  - Larger expert model (e.g., LLaMA 7B or 13B)
  - Enhanced RAG with hybrid search
- **Timeline**: 1-2 months
- **Improvements**: Better accuracy, more robust classification, improved response quality

### Large Effort:
- **Components**: All from medium effort, plus:
  - Comprehensive data collection and curation
  - Multi-task learning for handling multiple domains
  - Custom large language model training
  - Advanced RAG with multi-document reasoning and fact-checking
  - Human-in-the-loop evaluation and continuous learning
  - Scalable infrastructure for handling high traffic
  - Productionalized CI/CD pipeline for deployment, evaluation and fine-tuning
- **Timeline**: 3-6 months
- **Improvements**: State-of-the-art performance, ability to handle multiple domains, highly accurate and reliable responses, scalable system for production use

# Demo


https://github.com/user-attachments/assets/e2dec948-a1b5-46a7-9cb5-950c32063ec1

<img width="1393" alt="Screenshot 2024-10-06 at 21 07 11" src="https://github.com/user-attachments/assets/9a36aba3-8c12-4193-8430-8fa5238dc19f">
<img width="1405" alt="Screenshot 2024-10-06 at 21 07 18" src="https://github.com/user-attachments/assets/b9f228c9-1031-4002-bcf8-05be38b7b8c9">
