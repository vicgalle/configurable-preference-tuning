# Configurable Preference Tuning ⚙️📝

Code for the paper "Configurable Preference Tuning with Rubric-Guided Synthetic Data".

> tl;dr: Configurable Preference Tuning (CPT) uses rubric-guided synthetic data and DPO to enable LLMs to dynamically adjust behavior (e.g., writing style) at inference via system prompts.


## Structure

* `train.py`: fine-tuning code, using the preference data from the experimental section in the paper.

* `rubric_tamplates.py`: the rubrics used for the experiments in the paper.


## Datasets

The synthetic dataset used in the paper is released in the HugginfaceHub, under two different variants:

* [vicgalle/creative-rubrics-preferences](https://huggingface.co/datasets/vicgalle/creative-rubrics-preferences): this is the DPO-compatible version, in which the generations have been arranged into contrasting pairs.
* [vicgalle/creative-rubrics](https://huggingface.co/datasets/vicgalle/creative-rubrics): raw version, each row has the prompt, the rubric and the score target for the rubric, plus the response and the model used to generate it.

## Fine-tuned Models

Several CPT-tuned models are available in the HuggingFace Hub:

| Model              | Size  | Fine-tuned from |
| ------------------ | ----- | --------------- |
| [configurable-preference-qwen3-4b](https://huggingface.co/vicgalle/configurable-preference-qwen3-4b) | 4B    |    [Qwen3-4B](https://huggingface.co/unsloth/Qwen3-4B-unsloth-bnb-4bit)             |
| [configurable-preference-phi4](https://huggingface.co/vicgalle/configurable-preference-phi4) | 8.5B    |    [Phi-4](https://huggingface.co/unsloth/phi-4-unsloth-bnb-4bit)             |
| [configurable-preference-mistral-nemo-12b](https://huggingface.co/vicgalle/configurable-preference-mistral-nemo-12b) | 12B    |    [Mistral-Nemo-12B](https://huggingface.co/unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit)             |
| [configurable-preference-rocinante-12b](https://huggingface.co/vicgalle/configurable-preference-rocinante-12b) | 12B    |    [Rocinante-12B](https://huggingface.co/TheDrummer/Rocinante-12B-v1.1)            |
