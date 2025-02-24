# Test

- Important Notice: For fine-tuning the **OPT-175B** model, ensure that your system is equipped with at least `18GB of GPU memory` and `600GB of CPU memory`.

## Example: MeZO-SGD on OPT Models

```shell
# compare memory
bash test/mezo_sgd/hf_opt/test_memory_train.sh
```

```shell
# compare throughput
bash test/mezo_sgd/hf_opt/test_speed_train.sh
```

```shell
# compare accuracy
bash test/mezo_sgd/hf_opt/test_acc_train.sh
```

## Supported Tests

In progress...