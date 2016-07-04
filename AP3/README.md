This contains models 1-4, depending on what you (un)comment in the code.

Execute it with

```bash
$ ./basic_local_classifier.py --out ../DATA/out --hypes model-301/model-301.json
```


## Results

| Model | Features   | Topology                                     | PPR    | Mean Acc | Mean IoU | Fr. IoU | Time    | After-clearning        | CM                                                          |
| ----- | ---------- | -------------------------------------------- | ------ | -------- | -------- | ------- | ------- | ---------------------- | ----------------------------------------------------------- |
| 301   | 3 colors   | 3 x 64 (s, 0.5 d) x 64 (relu, 0.5 d) x 1 (s) | 92.88% |  0.6595  | 0.6125   | 0.8687  | 1.6486s | None                   | {0: {0: 38640407,  1: 408816}, 1: {0: 2654875, 1: 1303902}} |
| 302   | 3c + coords| 5 x 64 (s, 0.5 d) x 64 (relu, 0.5 d) x 1 (s) | 93.42% |  0.6971  | 0.6470   | 0.8792  | 1.7079s | None                   | {0: {0: 38567938, 1:  481285}, 1: {0: 2349644, 1: 1609133}} |
| 303   | 3c + coords| 5 x 64 (s, 0.5 d) x 64 (relu, 0.5 d) x 1 (s) | 93.51% |  0.7050  | 0.6537   | 0.8811  | 1.7128s | 3px erosion + dilation | {0: {0: 38541570, 1:  507653}, 1: {0: 2284099, 1: 1674678}} |
| 304   |            |                                              | 91.08% |  0.6641  | 0.5898   | 0.8492  | 2.3503s |                        | {0: {0: 37740157, 1: 1309066}, 1: {0: 2526770, 1: 1432007}} |
| 305   | 5 * 3colors|15 x 64 (s, 0.5 d) x 64 (relu, 0.5 d) x 1 (s) | 93.20% |  0.6898  | 0.6383   | 0.8758  | 3.9592s | None                   |



* s = sigmoid
* d = dropout
* Time: Avg Evaluation Time / Imag
