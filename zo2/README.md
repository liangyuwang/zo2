# Core code of ZO2

## Features

1. Fuse model dual-forward and optimizer step into model forward code. For example,

```python
# first-order one training step:
model.train()
loss = model(input, label)	# forward
loss.backward()		# backward
optimizer.step()	# update parameters, optimizer states

# zo2 one training step:
model.zo_train()	# Enable zo training
loss = model(input, label)	# fuse dual-forward, parameters and optimizer states updates
```

## Code Logic

1. Fuse model dual-forward and optimizer step into model forward code.

## In progress...