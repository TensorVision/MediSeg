This contains models 1-4, depending on what you (un)comment in the code.

Execute it with

```bash
$ time ./basic_local_classifier.py --data ../DATA
```


## Results

| Model | Features   | Topology                                     | PPR    | Mean Acc | Mean IoU | Fr. IoU | Time    | After-clearning        |
| ----- | ---------- | -------------------------------------------- | ------ | -------- | -------- | ------- | ------- | ---------------------- |
| 1     | 3 colors   | 3 x 64 (s, 0.5 d) x 64 (relu, 0.5 d) x 1 (s) | 95.33% |  0.8357  | 0.7471   | 0.9179  | 1.0010s | None                   |
| 2     | 3c+ coords | 5 x 64 (s, 0.5 d) x 64 (relu, 0.5 d) x 1 (s) | 95.51% |  0.8290  | 0.7499   | 0.9201  | 1.0979s | None                   |
| 3     | 3c+ coords | 5 x 64 (s, 0.5 d) x 64 (relu, 0.5 d) x 1 (s) | 95.49% |  0.8445  | 0.7555   | 0.9207  | 1.0768s | 3px erosion + dilation |

* s = sigmoid
* d = dropout
* Time: Avg Evaluation Time / Imag