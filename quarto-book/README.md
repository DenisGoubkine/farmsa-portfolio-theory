# Quarto Book

This folder contains the publishable web edition of the FARMSA portfolio theory project.

## Local Build

From this directory:

```bash
quarto render
```

If `quarto` is not on your shell path, use the installed user-space binary:

```bash
~/Library/Python/3.11/bin/quarto render
```

The rendered site is written to `_book/`.

## Vercel Deployment

The simplest deployment path is to publish the static `_book/` directory.

- Set the Vercel project root to `quarto-book` if you want Vercel to build from source.
- If Quarto is available in the build environment, use `quarto render` as the build command.
- If you want the fastest deployment path, deploy the already rendered `_book/` output as a static site.

## Structure

- `index.qmd`: front page and reading guide
- `chapters/`: publish-ready chapter copies of the notebooks
- `chapters/m6_optimizer.qmd`: web-specific documentation page for the optimizer interface
- `assets/report.css`: typography and layout styling
