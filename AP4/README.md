This model should have the same architecture as the one described in the
street segmentation paper (SST).

Execute it with

```bash
$ time ./sst_segmenter.py --data ../DATA
```

## Results

| Model  | Epochs  | Topology                        | PPR    | Mean Acc | Mean IoU | Fr. IoU | Time    | Comments                                             |
| ------ | ------- | --------------------------------| ------ | -------- | -------- | ------- | ------- | ---------------------------------------------------- |
| 401    | 200     | 10*10*3 : 64 3x3 sig filter : 1 | 89.44% |  0.5556  | 0.4983   | 0.8299  | 28.9672 | rmsprop, mse, batchsize=1024, patchsize=10, stride=1 |
| 402    | 1000    | 10*10*3 : 64 3x3 sig filter : 1 | 88.97% |  0.5397  | 0.4852   | 0.8240  | 18.0372 | rmsprop, mse, batchsize=1024, patchsize=10, stride=1 |
| 401-sst|         |                                 | 92.11% |                                         | {0: {0:  811688, 1: 3147089}, 1: {0:   244822, 1: 38804401}}
| 401-keras |      |                                 | 74.67% |                                         | make_equal, {0: {0: 3424316, 1: 534461},  1: {0: 10355404, 1: 28693819}}
