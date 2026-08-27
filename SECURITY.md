# Security policy

## Supported versions

Security fixes are accepted on the default branch of this repository.

## Reporting a vulnerability

Do not open a public issue for credentials, patient data, or exploitable bugs.

Please report privately via GitHub Security Advisories on this repository
(Security → Report a vulnerability), or contact the maintainers by email if
advisories are not enabled yet.

Include:

- a description of the issue and its impact;
- steps to reproduce, or a proof of concept that does not include live secrets;
- the affected commit or release if known.

You should receive an acknowledgement within a few working days.

## Secrets and health data

This project generates **synthetic** SCORE2 cohorts only. Never commit:

- `.env`, cloud keys, or database passwords;
- patient-level extracts (`data/`, `exp/`, CSV dumps).

## Publishing this repository

Git history of private clones may still contain warehouse extracts, notebooks,
or credentials that were later deleted from HEAD. **Do not push the existing
history to a public remote.** Create a fresh repository from the cleaned
working tree (orphan branch / squash) after a secret scan (`gitleaks` or
`trufflehog`).
