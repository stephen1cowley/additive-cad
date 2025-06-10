# Additive CAD

*As part of MEng project "Mitigating Hallucinations in LLMs", supervised by Dr Marcus Tomalin*

**arXiv paper:** (link coming soon)

Additive CAD is a proposed variant of Context-Aware Decoding (CAD) which adds the difference in probability distributions with and without context, as opposed to multiplying by their ratio. We achieve significant improvements on knowledge-conflict tasks (between prior LLM belief and provided contextual information) on the MemoTrap dataset, as shown below. Our research paper introduces a visual intuition for why Additive CAD is superior to regular CAD on specific knowledge-conflict examples.

Regular CAD:

<img src="https://github.com/user-attachments/assets/1ce39ba1-aac1-4ea5-823d-3f4b81fb3b38" style="width: 30%; height: auto;" alt="Screenshot 2025-05-28 114406">

Additive CAD:

<img src="https://github.com/user-attachments/assets/a5fcf981-9280-4861-b014-8edba90a1071" style="width: 30%; height: auto;" alt="Screenshot 2025-05-28 142308">


Our research also investigates the effect of incorporating the Decoding by Contrasting Layers (DoLa) method into the CAD inputs. We also explore the effect of varying the input prompts on CAD to address some of the inconsistent results seen across different papers when it comes to performance on the Natural Questions dataset.

## Evaluation code
The evaluation scripts in this repository can evaluate either CAD or Additive CAD on any json-format question-answer benchmark (json files for MemoTrap and Natural Questions are provided here). This might be useful for replicating some of the later results shown in the research paper, but note that the majority of the main results were evaluated on different scripts, provided [here](https://github.com/stephen1cowley/memotrap-testing) for academic purposes.

### To run
First install PyTorch from the official website. Then:
```
pip install numpy pandas transformers protobuf sentencepiece
```
Then run
```
python run_experiment.py --config path/to/config.json
```
See `src/experiment_types.py` for the required schema, which defines the hyperparameters of the desired experiment e.g. what dataset, what CAD/Additive CAD coefficients to use etc. Examples are provided in `experiment_config/`.
