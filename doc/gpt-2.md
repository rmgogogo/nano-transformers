# Nano GPT-2

## Paper

[2019 GPT-2: Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

![](image/gpt2.png)

- Major innovations
    - Move LayerNorm to earlier place, add one final LayerNorm after final self-attention
    - Initilize scale according to number of residual layers
    - Bigger
- Existing innovations used
    - Transformer, ResNet
    - Unsupervised Multi-task
    - Byte-level Byte Pair Encoding
    - GELU
    - Dropout

## Codes

Try the magic. It's better than GPT-1, even after smoke-mode training.

```
python gpt-2.py --sample
python gpt-2.py --plot
python gpt-2.py --smoke --predict
python gpt-2.py --train --predict
python gpt-2.py --predict --input "I don't want to"
```

## Samples

- Input: I want to
Output: I want to see you , " and he pointed to the <unk> , " I will not be able to get you to the <unk> .

- Input: Who is
Output: Who is this ? " asked Tom , as he saw the <unk> of the <unk> .

## Reference

- https://huggingface.co/docs/transformers/model_doc/openai-gpt#transformers.TFOpenAIGPTDoubleHeadsModel.call.example
- https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Tokenizer.example
- https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L157
