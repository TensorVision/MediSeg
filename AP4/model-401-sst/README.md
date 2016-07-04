```bash
sst train --hypes model-401-sst.json
sst test --hypes model-401-sst.json --out ../../DATA/out
```

Results:

17x17: Eval results: {0: {0: 2491123, 1: 1467654}, 1: {0: 1323298, 1: 37725925}}, Accurity: 0.935106212798
51x51: Eval results: {0: {0: 2578117, 1: 1380660}, 1: {0: 1202032, 1: 37847191}}, Accurity: 0.939948567708