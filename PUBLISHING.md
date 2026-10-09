# Create the independent EXaCTz repository

The remote repository has not been created. This environment has no GitHub CLI and its available GitHub connector does not expose repository creation.

For author review, create an empty **private** repository named EXaCTz under the intended GitHub account or organization. Do not initialize it with a README, .gitignore, or license; this candidate already contains the first two. Copy its repository URL into the command below.

Run from the candidate directory, not the original ExtreM_z workspace:

```bash
cd /Users/yuxiaoli/Desktop/ExtreM_z/release/EXaCTz-0.1.0-rc1
git init -b main
git add .
git commit -m "Prepare EXaCTz single-GPU and multi-GPU source candidate"
git remote add origin https://github.com/OWNER/EXaCTz.git
git push -u origin main
```

Replace OWNER with the chosen account or organization. GitHub credentials and permission to that repository are required. For a GitHub CLI environment, an alternative after the commit is `gh repo create OWNER/EXaCTz --private --source=. --remote=origin --push` (use this instead of the last two Git commands).

Before making the repository public, resolve the license and target-system validation items in RELEASE_REVIEW.md. Record validated dependency versions and build/run evidence in the repository. Then select the final version/tag and create a GitHub release with source and checksum assets. The 0.1.0-rc1 name is a proposed candidate identifier, not an existing Git tag or public release.

Suggested reply to the advisor:

> I have prepared a source release candidate for EXaCTz with both the single-GPU and multi-GPU implementations, build instructions, and run examples. I am organizing it as a separate EXaCTz GitHub repository. Before the public release, I still need to confirm the license and complete clean-build and runtime validation on an NVIDIA GPU system.
