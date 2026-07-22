# Security Policy

## Supported versions

Until Open Evals reaches 1.0, security fixes are made on the latest code on the `main` branch.
Older commits and development snapshots are not supported.

## Reporting a vulnerability

Please do not disclose a suspected vulnerability in a public issue, discussion, or pull request.

Use GitHub's **Report a vulnerability** option on the repository's Security tab. If private
vulnerability reporting has not yet been enabled, contact the repository owner privately through
a method listed on their GitHub profile and ask for a secure reporting channel. Do not send
secrets or exploit details until a private channel is established.

Include, when possible:

- the affected version or commit;
- the impact and conditions required to trigger the issue;
- minimal reproduction steps or a proof of concept;
- any suggested mitigation;
- whether the issue is already public or known to others.

A maintainer should acknowledge a complete report within seven days. Timelines for validation,
fixes, releases, and coordinated disclosure depend on severity and complexity and will be agreed
with the reporter.

## Scope notes

Open Evals sends configured prompts, images, bearer tokens, and model options to the configured
inference endpoint. Users are responsible for trusting that endpoint and protecting benchmark
configuration and result files. Reports about vulnerabilities in third-party inference servers,
models, drivers, or `nvidia-smi` should be directed to those projects unless Open Evals creates or
amplifies the issue.
