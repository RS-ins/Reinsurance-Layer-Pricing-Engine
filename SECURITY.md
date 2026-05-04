# Security Policy

## Project Scope

The Reinsurance Layer Pricing Engine is an **educational and research tool**. It is a standalone Python library and static web dashboard with no server-side components, no authentication, no database, and no network-facing services.

Given this scope, the attack surface is very limited. However, responsible disclosure of any security concerns is still welcomed.

---

## Supported Versions

| Version | Supported |
|---|---|
| Latest (`main` branch) | ✅ Yes |
| Older commits | ❌ No |

Only the current state of the `main` branch is actively maintained.

---

## Reporting a Vulnerability

If you discover a security issue in this project, please report it responsibly:

**Preferred:** Open a [GitHub Issue](https://github.com/RS-ins/Reinsurance-Layer-Pricing-Engine/issues) with the label `security`. For sensitive issues, use GitHub's private vulnerability reporting feature under the **Security** tab of the repository.

Please include:
- A description of the vulnerability
- Steps to reproduce it
- The potential impact
- Any suggested fix if you have one

---

## What Counts as a Security Issue

Given the educational nature of this project, relevant security concerns would include:

- **Dependency vulnerabilities** — a known CVE in `numpy`, `scipy`, `matplotlib`, `plotly`, or other dependencies
- **Malicious input handling** — if the library could be exploited when processing untrusted input (e.g. if used as part of a larger application)
- **Dashboard issues** — any XSS or content injection vulnerability in `app/dashboard.html`
- **Supply chain issues** — any concern about the integrity of the codebase or build process

---

## What Does Not Apply

- This project has no user accounts, passwords, or authentication
- It stores no personal data
- It has no server-side execution in production (the dashboard is a static HTML file)
- It is not intended for use in production systems without independent review

---

## Disclaimer

This project is **not** intended for production use, commercial pricing, or regulatory filings. See the [README](README.md) for the full disclaimer. Users who deploy this software in any professional context do so at their own risk and are responsible for their own security review.
