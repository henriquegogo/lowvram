# Paper

Draft: *lowvram: Transparent GPU Memory Oversubscription for Unmodified CUDA Programs*.

## Build

```sh
make                 # latexmk; or: TEX=tectonic make
```

The preamble falls back to plain LaTeX when `microtype`, `booktabs` or `xcolor` are
missing; install `texlive-latex-extra` for the intended typesetting. Missing figures
or numbers render as **??** instead of failing the build, so the draft builds at any
stage of a benchmark run.

## Where the content comes from

| File | Origin |
|---|---|
| `paper.tex`, `references.bib` | written by hand |
| `figures/*.pdf`, `figures/*.png` | `../bench/plot.py` (PDF for the paper, PNG for blog and slides) |
| `generated/numbers.tex` | `../bench/plot.py`: one `\newcommand` per number quoted in the text |
| `generated/table_*.tex` | `../bench/plot.py` |

Quote a measured value through its macro (`\num{yueLvDecode}`), never as a literal, so
that a new run updates the text. To add a number, add it to `tables()` in `plot.py`;
macro names must be letters only.

## Status

- All sections drafted; results written from the October 2026 run.
- `\todo{}` marks text that still waits for results or a decision.
- References marked `VERIFY` in `references.bib` were written from memory.
