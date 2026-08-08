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

## Author decisions required before Git initialization and remote creation

- [ ] Confirm repository name.
- [ ] Confirm initial visibility (taskbook default: private).
- [ ] Select and approve a public code license.
- [ ] Confirm final contact details and citation metadata.
- [ ] Confirm whether the static site will use the same repository or a separate deployment repository.

## Remote release

- [ ] Show the exact `git init`, add, status, commit, and `gh repo create` commands to the author.
- [ ] Initialize Git only inside this directory after approval.
- [ ] Inspect staged files before commit; never use `git add -f`.
- [ ] Create the private remote and verify branch, remote, and web contents.
- [ ] Create a tested pre-release/release tag.
- [ ] Connect Zenodo, archive the approved release, and record the real DOI.
- [ ] Update citation, version page, and data-availability statement with real identifiers.
- [ ] Deploy the static site and test the public URL in desktop, mobile-width, and private browsing contexts.
