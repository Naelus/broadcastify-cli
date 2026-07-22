# UI and platform decisions

## Why WinUI 3 on Windows

WinUI 3 is the native Windows reference because this was a new application, not
an existing WinForms codebase. It provides current Fluent controls, DPI behavior,
Windows App SDK lifecycle, media integration, and modern navigation while
remaining a native desktop application rather than a hosted browser shell.

WinForms remains a supported and productive framework, but it would optimize for
implementation simplicity over the contemporary native layout/navigation model
this application needs. WPF would also be viable, but WinUI is Microsoft's
current direction for new native Windows applications.

The desktop is unpackaged x64 today. That avoids requiring MSIX during
development, but the verified publish still depends on the repository Python
environment and is not a complete consumer installer. See
[Windows publish](../windows-publish.md).

## Why a separate browser companion

The Python backend already owns acquisition, processing, persistence, and
analysis. A responsive browser client provides the same Library, New Archive,
Review, Area Watch, and Settings contract on Linux, macOS, TrueNAS, and Windows
without pretending WinUI is portable.

This keeps platform roles clear:

- WinUI is the polished Windows shell.
- The browser UI is the cross-platform and headless/NAS surface.
- The CLI is the automation and diagnostic surface.

The browser does not replace the native app on Windows, and WinUI does not force
other platforms into a Windows abstraction.

## Navigation model

Both UIs are task-oriented rather than one long scrolling form:

- **Local Library** for retained state and continuation
- **New Archive** for discovery and acquisition
- **Review & Ask** for incidents, clips, summaries, and questions
- **Area Watch** for profiles, queues, and regional leads
- **Settings** for Setup, Processing, Analysis & AI, and Account

Master/detail layouts keep a bounded list and one selected evidence package
visible. Generated prose is secondary to coverage and citations.

## Local and LAN hosting

The browser binds to loopback by default. Explicit private/link-local binding or
`0.0.0.0` enables trusted-LAN access. Anyone who can reach that listener can
receive a session and read retained transcripts/audio or start supported jobs;
the cookie/action-token and same-origin checks are CSRF/session controls, not
user authentication.

LAN mode is therefore appropriate only for a trusted network. It must not be
port-forwarded or publicly proxied without real authentication and TLS. See
[Web UI](../web-ui.md).

## Current portability boundary

Real Windows, Linux, and TrueNAS paths are validated. Apple Metal support is
implemented but awaits a real Mac workflow. Cross-platform keychain integration
and non-developer packaging remain open in
[GitHub Issues](https://github.com/Naelus/broadcastify-cli/issues).
