# Proposed remote-creation commands — not yet executed

Replace `<repository-name>` only after author confirmation. Initial visibility follows the taskbook default of private unless the author explicitly chooses otherwise.

```powershell
git init -b main
git add .
git status --short
git diff --cached --stat
git commit -m "Prepare initial COMPASS reproducibility package"
gh repo create LiaoZitong/<repository-name> --private --source . --remote origin --push
git remote -v
git branch --show-current
```

Before `git commit`, inspect the full staged list for raw/processed data, compressed databases, office documents, credentials, local paths, review material, and logs. Do not use force-add or force-push.

