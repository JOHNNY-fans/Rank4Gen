# Rank4Gen

## Introduction

Welcome to **Rank4Gen** 🎉  
We propose **Rank4Gen**: *RAG-Preference-Aligned Document Set Selection and Ranking*.

<div align="center">
  <img src="figure/framework.svg">
</div>

---

## 📦 Installation

Clone this repository and install dependencies:

```bash
git clone git@github.com:JOHNNY-fans/Rank4Gen.git
cd Rank4Gen
pip install -r requirements.txt
````

---

## 📁 Repository Structure

```
Rank4Gen/
  code/                         # training scripts (SFT / DPO)
  evaluation/
    inference.py                # ranker inference (select & rank docs)
    downstream_model_test.py    # downstream QA inference (RAG answering)
    evaluate.py                 # evaluation metrics (EM/F1)
    input/
      sampled_data.jsonl        # example test file (provided)
    selected_results/           # ranker-selected docs output
    downstream_results/         # downstream QA outputs
    model_descriptions/         # downstream model preference descriptions
  figure/                       
  LICENSE
  README.md
  requirements.txt
```

---

## 🚀 Evaluation Pipeline (Quick Start)

The evaluation pipeline consists of **three steps**:

1. **Run the Ranker** to select a subset of documents for each query
2. **Run the downstream generator** to answer the query based on selected docs
3. **Compute EM/F1** based on downstream answers

We provide an example dataset at:

```
evaluation/input/sampled_data.jsonl
```

---

## 0️⃣ Start an OpenAI-Compatible Server (vLLM)

Both the ranker inference script and the downstream QA script assume an **OpenAI-compatible** endpoint (`/v1/chat/completions`).
You can host your model with vLLM like this:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Rank4Gen \
  --served-model-name Rank4Gen \
  --port 8800
```

Then your API base URL is:

```text
http://127.0.0.1:8800/v1
```

> If you run ranker and downstream model on different servers, start two vLLM servers with different `--port` and `--served-model-name`.

---

## 1️⃣ Ranker Inference (Document Selection & Ranking)

This step reads the test dataset (`jsonl`) and calls your deployed **Ranker** model to output selected document ids, then maps them back to original document texts.

### Input (example)

Each line in `evaluation/input/sampled_data.jsonl` follows:

```json
{
  "id": "...",
  "query": "...",
  "answers": ["..."],
  "documents": [
    {"id": 1, "text": "..."},
    {"id": 2, "text": "..."}
  ]
}
```

### Run

```bash
python evaluation/inference.py \
  --input evaluation/input/sampled_data.jsonl \
  --lang en \
  --mode index|snapshot \
  --tokenizer_path /path/to/ranker_tokenizer_or_model \
  --model_name Rank4Gen \
  --api_url http://127.0.0.1:8800/v1 \
  --downstream_model default \
  --output_path evaluation/selected_results/sampled_data.selected.jsonl
```

### Output (downstream-ready)

`evaluation/selected_results/sampled_data.selected.jsonl` will contain:

```json
{
  "id": "...",
  "query": "...",
  "std_answer": "...",
  "documents": [
    {"id": 3, "text": "..."},
    {"id": 7, "text": "..."}
  ],
  "answer": ""
}
```

---

## 2️⃣ Downstream QA Inference (Answer Generation)

This step calls the downstream generator to produce answers conditioned on the **selected documents**.

### Run

```bash
python evaluation/downstream_model_test.py \
  --data_path evaluation/selected_results/sampled_data.selected.jsonl \
  --output_dir evaluation/downstream_results \
  --dataset_name sampled_data \
  --model_name downstream_model_name \
  --api_endpoint http://127.0.0.1:8800 \
  --model_path /path/to/downstream_hf_model
```

### Output

The script writes a JSONL file to:

```
evaluation/downstream_results/sampled_data-downstream_model_name-all.jsonl
```

Each line includes:

* the final prompt used (`final_prompt`)
* raw model output (`raw-response`)
* parsed `answer`
* status (`success`, `failed_parsing`, `failed_all_attempts`)

---

## 3️⃣ Evaluate EM/F1 (Answer Quality)

Finally, compute EM/F1 on downstream answers:

```bash
python evaluation/evaluate.py \
  --result_file evaluation/downstream_results/sampled_data-downstream_model_name-all.jsonl \
  --ranker Rank4Gen \
  --method index \
  --downstream_model downstream_model_name \
  --dataset_name sampled_data
```

### Output Format (Console Table)

The evaluation script prints a compact table:

* Ranker
* Method
* Downstream_Model
* DatasetName
* EM / F1

---

## 🧪 Training Scripts (SFT / DPO)

Training scripts are provided under:

```
code/
  SFT/
  DPO/
```

Each folder contains runnable `.sh` scripts to reproduce training.

---

## 📄 License

This project is released under the license in `LICENSE`.

---

## 📚 Citation

If you use Rank4Gen in your research, please cite:

```bibtex
@article{rank4gen2026,
  title={Rank4Gen: RAG-Preference-Aligned Document Set Selection and Ranking},
  author={...},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2026}
}