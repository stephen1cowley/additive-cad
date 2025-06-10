# Additive CAD

arXiv paper: (link coming soon)

Additive CAD is a variant of Context-Aware Decoding (CAD) which adds the difference in probability distributions with and without context, as opposed to multiplying by their ratio. We achieve significant improvements on knowledge-conflict tasks (between prior LLM belief and provided contextual information) on the MemoTrap dataset, as seen below. Our research paper introduces a visual intuition for the particular knowledge-conflict scenarios where Additive CAD is superior to regular CAD.

Regular CAD:

<img src="https://github.com/user-attachments/assets/1ce39ba1-aac1-4ea5-823d-3f4b81fb3b38" style="width: 30%; height: auto;" alt="Screenshot 2025-05-28 114406">

Additive CAD:

<img src="https://github.com/user-attachments/assets/a5fcf981-9280-4861-b014-8edba90a1071" style="width: 30%; height: auto;" alt="Screenshot 2025-05-28 142308">


Our research also investigates the effect of incorporating the Decoding by Contrasting Layers (DoLa) method into the CAD inputs. We also explore the effect of prompt on CAD to address some of the inconsistent results seen across different papers when it comes to CAD performance on the Natural Questions dataset.

## Evaluation Code
This repository contains some of the evaluation code used toward the latter parts of the research project. Additional evaluation code used for the majority of the results can be found in [another repository](https://github.com/stephen1cowley/memotrap-testing).

## To run an experiment
You will first need to install torch from the official website. Then:
```
pip install numpy pandas transformers protobuf sentencepiece
```

