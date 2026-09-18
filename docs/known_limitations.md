# Known limitations

- This is a compact code package. Large public and provider-hosted sources, including EPA ECOTOX, remain available through their official services and are not mirrored here.
- Provider-specific acquisition and preparation are outside the core code subset. Full-data reconstruction starts from the documented official sources and requires preparing inputs that meet the frozen contracts.
- The original analysis workspace has no release Git history; the R1 code release is identified by its Git tag/commit, SHA-256 manifests, and recorded source versions.
- Base Figures 1–5 and R1 replacements for Figures 2, 3, and 5 are included. R1 SI Figures S3, S4, S7, and S8 are included, but S3 retains two derived base-SI source-table dependencies documented in its script; DOCX/SI assembly remains author-controlled.
- Figure 5 requires a separately downloaded Census boundary archive.
- Whole-chemical holdout is an internal generalization assessment within the analyzed corpus. External or prospective validation will require a new, independently frozen evidence set.
- Profile-likelihood intervals address record-censoring and working-SSD uncertainty as specified. MNAR scenarios are sensitivity bounds because the missingness mechanism is not identified by the observed data.
- Local occurrence and NAS information are species-relevance signals. Broader ecological deployability, current local status, and method validation require follow-up evidence.
- The public repository uses an all-rights-reserved source-availability notice, not an open-source licence. No Zenodo or other archival DOI is claimed. Author ORCIDs remain omitted until individually confirmed.
