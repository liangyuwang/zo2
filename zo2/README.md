# Core code of ZO2

## Features

1. Fuse model dual-forward and optimizer step into model forward code. For example,

```python
# first-order one training step:
loss = model(input, label)	# forward
loss.backward()		# backward
optimizer.step()	# update parameters, optimizer states

# zo2 one training step:
model.zo_training = True	# Enable zo training
loss = model(input, label)	# fuse dual-forward, parameters and optimizer states updates
```

2. 

## Code Logic

1. Fuse model dual-forward and optimizer step into model forward code.

## TODO

[] Huggingface models

    [] GPT2

    [] OPT

    [] Llama

[] Huggingface trainer
