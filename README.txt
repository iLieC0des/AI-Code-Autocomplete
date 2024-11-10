>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> AI-POWERED CODE AUTOCOMPLETE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Overview
This project implements an AI-powered code autocomplete tool using the GPT-2 language model, fine-tuned on a dataset of code snippets to generate relevant code suggestions and completions.

# Key Features
- **Code Autocompletion**: Suggests code completions based on partial code input.
- **Custom Fine-Tuning**: Fine-tuned GPT-2 model improves code generation for programming-related tasks.
- **Local Testing**: Run locally for testing and evaluation.

# Requirements
Install the following dependencies:
```plaintext
transformers==4.33.1
datasets==2.11.0
torch==2.1.0
numpy==1.24.2
pandas==1.5.3


Setup-

# ProjectDOD Code Autocomplete Tool

This project fine-tunes a GPT-2 model for code autocompletion using the Hugging Face Transformers library.

# Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/projectDOD.git
   cd projectDOD
   ```

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Pretrained Model and Dataset**
   - In your code, use `transformers` to automatically download the GPT-2 model and datasets on first use, eliminating the need to store them in the repository.

# Running the Project

Run the script:
```bash
python projectDOD.py

IMPORTANT!

1 Project Setup and Installation
Clone the Repository. 

git clone https://github.com/yourusername/projectDOD.git
cd projectDOD

2. Set up a Virtual Environment (recommended)

python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt


4. Model and Dataset Download
The script automatically downloads the GPT-2 model and dataset on first run using transformers.

5. Running the Project
python projectDOD.py
Note: Ensure you are running this on the Anaconda prompt or the environment you created.



# Models Overview

This section explains the structure, configuration, and usage of the GPT-2 model used in the project. You can modify and fine-tune the model based on your needs.

# Scripts or Config Files Related to Models:

# Model Fine-Tuning Script:

The script responsible for fine-tuning the GPT-2 model on a custom dataset is `projectDOD.py`. This script performs the following:

- **Loads a pre-trained GPT-2 model**: The model is loaded using the `transformers` library from Hugging Face.
- **Loads and preprocesses the dataset**: The dataset used for fine-tuning is the `codeparrot/codeparrot-clean` dataset.
- **Tokenizes the dataset**: The dataset is tokenized for feeding into the model.
- **Fine-tunes the GPT-2 model**: The script uses the Hugging Face `Trainer` to fine-tune the model.
- **Saves the trained model and tokenizer locally**: After training, the model and tokenizer are saved locally for later use.

# Configuration Files:
No external configuration files are required for this project since all model parameters and training configurations are directly defined within `projectDOD.py`. However, you can tweak the following parameters as needed:

- **Learning Rate**: Set in the `TrainingArguments` (e.g., `learning_rate=5e-5`).
- **Batch Size**: Defined in `TrainingArguments` (e.g., `per_device_train_batch_size=4` and `per_device_eval_batch_size=8`).
- **Number of Epochs**: Defined in `TrainingArguments` (e.g., `num_train_epochs=3`).

These settings can be adjusted based on available computational resources or project needs.

#Saving and Loading the Model:

After training, the model and tokenizer are saved locally using the following methods:

- **Model**: `model.save_pretrained("path/to/save/model")`
- **Tokenizer**: `tokenizer.save_pretrained("path/to/save/tokenizer")`

You can load these saved models later for inference or further fine-tuning.

