# Release checklist

## Local package

- [x] Source workspace audited without restructuring.
- [x] Core stages copied to an independent sibling directory.
- [x] Real data and manuscript/review material excluded.
- [x] Package-relative runner and figure paths prepared.
- [x] Dependency, seed, input, output, and site-export contracts documented.
- [x] Numerical checkpoints and cross-package validation implemented.
- [x] All local validation commands pass on the final package snapshot.
- [x] Large-file, secret, personal-path, and prohibited-marker scan passes.
- [x] SHA-256 source and key-output manifests are generated; regenerate package checksums after any later edit.

## Repository and deployment decisions

- [x] Repository name confirmed as `COMPASS`.
- [x] Initial repository visibility confirmed as private.
- [x] Public source-availability terms selected: publicly viewable, all rights reserved; no open-source reuse grant is implied.
- [x] Citation metadata synchronized to the author list and version 1.0.1.
- [x] Static site uses a separate Sites deployment with public access.

## Remote release

- [x] Git repository initialized only inside this independent package after approval.
- [x] Inspect staged files before commit; never use `git add -f`.
- [x] Private `LiaoZitong/COMPASS` remote, `main` branch, and web contents verified.
- [x] Create the tested `v1.0.1` release tag after the final validation commit.
- [x] Record that no Zenodo deposition or DOI is claimed; a future deposition requires a separate author decision.
- [x] Update citation, version page, and data-availability statement with the verifiable GitHub repository and release tag.
- [x] Public companion site configured; each update is tested locally in desktop, mobile-width, interaction, and private contexts before deployment.
