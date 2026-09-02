# Security and release hygiene

MAGE calls external model providers and can connect to Neo4j. Treat all credentials and model
responses as private research infrastructure.

## Local setup

1. Copy `.env.example` to `.env`.
2. Fill credentials locally or inject them through the process environment.
3. Keep inputs, uploads, generated graphs and logs outside the repository (the default `.gitignore`
   already excludes the common locations).

## Pre-publish checks

From the MAGE directory, run:

```powershell
rg -n -i --hidden --glob '!*.pyc' --glob '!__pycache__/**' `
  '(sk-[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{20,}|-----BEGIN (RSA|OPENSSH|PRIVATE) KEY-----|bearer\s+[A-Za-z0-9._-]{20,}|(api[_-]?key|password|secret|token)\s*[:=]\s*["''][^"'']{8,})' .
```

The command should return no real credential values. Variable names such as `OPENAI_API_KEY` are
expected; their values must be empty or placeholder text in public files. Also inspect:

```powershell
git status --short
git diff -- . ':!*.png' ':!*.jpg' ':!*.jpeg'
```

If a credential was ever committed, revoke or rotate it before publishing and remove it from the
repository history; deleting the current file alone is not sufficient.

## Reporting a vulnerability

Do not open a public issue containing a credential, private article, model response or database
address. Contact the repository maintainers through the private channel listed on the GitHub
repository once it is public.

