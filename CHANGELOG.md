# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `LICENSE` file (MIT), now shipped in the wheel and sdist.
- CI test workflow: ruff, mypy (strict) and pytest on Python 3.10-3.13 for every
  push and pull request. Publishing to PyPI now runs the same checks first.
- Install section in the README.

## [0.1.0] - 2026-09-22

First public release.

- 373/373 equations of ANSI/AISC 360-16 implemented across Chapters B to K plus
  Appendices 1-4 and 6-8.
- LRFD and ASD from one code path.
- Every function cites its source: equation number, section and Specification page.
- 1,138 tests, including checks against the AISC Design Examples V16.0.
- No runtime dependencies, so it imports inside FreeCAD's embedded Python.

[Unreleased]: https://github.com/ahmedalisteel-star/pyaisc360/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ahmedalisteel-star/pyaisc360/releases/tag/v0.1.0
