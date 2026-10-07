# Submission release checklist

Use this checklist only after the manuscript and consolidated figures are frozen.

## Scientific freeze
- [x] Gate A: external GeoLife validation passed and committed.
- [x] Gate B: SmolLM2 second-backbone replication passed and committed.
- [x] Main manuscript reduced to the five-figure causal story.
- [x] Extended neuro-inspired analyses preserved in Supplementary Results.
- [x] Main-text bibliography verified and committed.
- [ ] Latest CI green on Python 3.11 and 3.12.
- [ ] Consolidated Figures 1–5 regenerated from committed JSON and visually inspected.

## Repository freeze
- [ ] Run: `python -m src.eval.make_submission_figures`.
- [ ] Confirm `paper/figures/Fig1_...Fig5_*.svg/pdf` exist.
- [ ] Confirm `git status` clean.
- [ ] Confirm every quantitative main-text claim points to a committed JSON.
- [ ] Update `CITATION.cff` from `1.0.0-rc1` to `1.0.0`.
- [ ] Create annotated tag:
      `git tag -a paper-v1.0.0 -m "Spatial-LLM submission artifact"`
- [ ] Push tag:
      `git push origin paper-v1.0.0`
- [ ] Create GitHub Release from that exact tag.

## Zenodo — requires repository owner action
- [ ] Sign in to Zenodo.
- [ ] Connect the GitHub account and enable archival for `Mohammadzamanid/Spatial-LLM`.
- [ ] Publish/archive the GitHub Release.
- [ ] Copy the minted DOI into `CITATION.cff`, README, and manuscript Data/Code Availability section.
- [ ] Commit the DOI-only metadata update without changing scientific artifacts.
- [ ] If Zenodo creates a version DOI and concept DOI, cite the version DOI for the exact submission artifact and optionally expose the concept DOI in README.

## Submission packet
- [ ] Main manuscript.
- [ ] Supplementary Results.
- [ ] Figures 1–5 as vector PDF/SVG or journal-required format.
- [ ] Source-data/result JSONs.
- [ ] Code/data availability statement with DOI.
- [ ] Cover letter.
- [ ] Reporting checklist / editorial policy forms for the selected journal.
