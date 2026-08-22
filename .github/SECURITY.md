# Security Policy

## Supported Versions

meowmeowcatcam is actively maintained. Security fixes are applied to the latest release on the `main` branch. Older tagged releases are not backported.

| Version         | Supported          |
| --------------- | ------------------ |
| Latest (`main`) | :white_check_mark: |
| Older releases   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in meowmeowcatcam, please **do not open a public GitHub issue**. Instead, report it privately using one of the following methods:

- **GitHub Private Vulnerability Reporting**: Go to the repository's **Security** tab → **Report a vulnerability**. This is the preferred method, as it keeps the report confidential until a fix is released.
- **Email**: If private reporting is unavailable, contact the maintainer directly (see the GitHub profile for contact info) with details of the issue.

Please include as much of the following as possible:

- A description of the vulnerability and its potential impact
- Steps to reproduce (proof-of-concept code, logs, or screenshots if applicable)
- The affected component (e.g., download engine, settings handling, GUI, network requests)
- Your suggested severity, if you have one

### What to Expect

- **Acknowledgment**: within 3–5 business days of your report.
- **Status updates**: as the issue is triaged, reproduced, and worked on.
- **Disclosure**: once a fix is available, a security advisory will be published on GitHub and credit given to the reporter (unless you prefer to remain anonymous).

Please allow a reasonable amount of time for a fix to be developed and released before any public disclosure.

## Scope

meowmeowcatcam is a desktop application that downloads and processes media using `yt-dlp`, `ffmpeg`, and `aria2c` as external binaries. Security concerns particularly relevant to this project include:

- **Command injection** via filenames, URLs, or arguments passed to `yt-dlp`, `ffmpeg`, `aria2c`, or Deno subprocess calls
- **Path traversal** in output filenames or download destinations
- **Unsafe deserialization** or parsing of `settings.json` or other config/cache files
- **Insecure handling of external binaries** (e.g., loading binaries from untrusted paths, missing integrity checks on downloaded tools)
- **Dependency vulnerabilities** in `yt-dlp`, `requests`, `PyQt5`, or `qtawesome`
- **SSRF or unsafe network requests** when fetching thumbnails, metadata, or update checks

Vulnerabilities in third-party dependencies (`yt-dlp`, `ffmpeg`, `aria2c`, `PyQt5`, etc.) should generally be reported upstream to those projects, but feel free to flag them here as well if they materially affect meowmeowcatcam's default configuration.

## Best Practices for Users

- Always download meowmeowcatcam and its bundled/recommended binaries (ffmpeg, aria2c, Deno) from official sources.
- Keep dependencies up to date (`pip install -r requirements.txt --upgrade`).
- Avoid running the application with elevated/administrator privileges unless required.
- Review `settings.json` and any downloaded content paths before running batch operations.

Thank you for helping keep meowmeowcatcam and its users safe.
