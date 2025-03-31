# LoRA-fine-tuning-medical-assistant-based-on-llama3

## Quick Start
### installing dependencies
```bash
conda create -n med_assistant_llama python=3.12.9
conda activate med_assistant_llama

git clone https://github.com/Minijie3/LoRA-fine-tuning-medical-assistant-based-on-llama3
unzip dataset.zip  # it's recommended to put dataset in the same directory as the project: ./datasets/transformed_data.json

pip install -r requirements.txt
```

### Getting the llama3-8b-instruct
```bash
# Try this if it's hard to install the llama3-8b-instruct from huggingface.co
export HF_ENDPOINT=https://hf_mirror.com

python base_model.py
```

### Fine-tuning the model
```bash
python train.py
```

### Usage
The model can be fine-tuned on a L40S GPU for 16 hours with a batch size of 4.

### Evaluation
```bash
streamlit run merge_streamlit.py
