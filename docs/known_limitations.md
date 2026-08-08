# Known limitations

- This is a compact code package. Large public and provider-hosted sources, including EPA ECOTOX, remain available through their official services and are not mirrored here.
- Provider-specific acquisition and preparation are outside the core code subset. Full-data reconstruction starts from the documented official sources and requires preparing inputs that meet the frozen contracts.
- The source workspace has no valid Git history; the freeze is identified by SHA-256 manifests and recorded versions/dates.
- Main Figures 1–5 are included. SI Figure S1–S7 assembly remains coupled to the author-controlled submission package and is not included in this core subset.
- Figure 5 requires a separately downloaded Census boundary archive.
- Whole-chemical holdout is an internal generalization assessment within the analyzed corpus. External or prospective validation will require a new, independently frozen evidence set.
- Profile-likelihood intervals address record-censoring and working-SSD uncertainty as specified. MNAR scenarios are sensitivity bounds because the missingness mechanism is not identified by the observed data.
- Local occurrence and NAS information are species-relevance signals. Broader ecological deployability, current local status, and method validation require follow-up evidence.
- The public license, final repository URL, release tag, DOI, author ORCIDs, and archival data statement await author confirmation.
