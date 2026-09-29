# Exercises

Every page of the course has one or more drills here. Each drill has three files:

| File | What it is |
|---|---|
| `drills/<name>.py` | Function signatures and docstrings. **You** fill in the bodies. |
| `solutions/<name>.py` | Reference implementation. Don't open it until you have a passing attempt or are truly stuck. |
| `tests/test_<name>.py` | Tests. By default they run against your code in `drills/`. |

```bash
pip install -r exercises/requirements.txt       # CPU torch is enough

pytest exercises/tests/test_attention.py         # test YOUR implementation
pytest exercises --solutions                     # sanity-check every reference solution
LLM_SOLUTIONS=1 pytest exercises                 # same, via env var
```

The tests are self-contained: they compare against PyTorch built-ins or check mathematical properties, so they never import the solutions.

Interview practice tip: time-box each drill to what an interview gives you (usually 20 to 40 minutes), write it without looking anything up, and only then run the tests.

## Index

| Page | Drill | File |
|---|---|---|
| 07 | GQA with a KV cache; MHA→GQA mean-pool conversion | `drills/gqa.py` |
| 07 | MLA: decompressed forward and absorbed decode; KV bytes per token | `drills/mla.py` |
