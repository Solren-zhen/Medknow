# Security Policy

MedKnow is a research/education project, not a medical device. We take security
and data-privacy reports seriously.

## Reporting a vulnerability

Please **do not open a public GitHub issue** for security problems. Instead,
report them privately via **GitHub's private vulnerability reporting**:

1. Go to the repository's **Security** tab → **Report a vulnerability**.
2. Describe the issue, including:
   - affected file / function and version,
   - a minimal reproduction,
   - the impact you believe it has.

We aim to acknowledge reports within 5 business days and to respond with a
remediation plan or an assessment within 30 days.

## Scope

In scope:

- Code execution or data-exposure vulnerabilities in `scripts/`, `src/medknow/`,
  `api/`, or the demo app.
- Prompt-injection or model-abuse risks in the demo that could mislead users.

Out of scope:

- Theoretical model weaknesses without a concrete exploit path (these are
  research findings — open an issue instead).
- Vulnerabilities in upstream dependencies; please report them to the
  respective projects.

## Supported versions

| Version | Supported |
|---|---|
| main (development) | ✅ |
| 1.0.x | ✅ |
