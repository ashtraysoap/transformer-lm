### Transformer language model for text generation

Code is heavily based on OpenAI's [gpt-2 code samples](https://github.com/openai/gpt-2) and their original [transformer language model](https://github.com/openai/finetune-transformer-lm)

The initial codebase lacks code for training and the latter is oriented towards finetuning the model for supervised tasks. This repository aims to provide an implementation allowing straightforward text generation from the model.

#### Requirements
- numpy
- tensorflow>=1.12.0

#### Training
```
python3 train.py path/to/text/corpus
```

#### Sampling
```
python3 sample.py path/to/model/checkpoint path/to/vocab path/to/hyperparams
```

See source for additional options.
