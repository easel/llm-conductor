# Release Process

This document describes how to create and publish releases for llm-conductor.

## Prerequisites

### 1. Configure GitHub Pages

1. Go to repository **Settings** → **Pages**
2. Set **Source** to: `Deploy from a branch`
3. Select branch: `gh-pages`
4. Select folder: `/ (root)`
5. Click **Save**

Your package index will be available at: https://easel.github.io/llm-conductor/

### 2. Configure PyPI Publishing

**Option A: Trusted Publishing (Recommended)**

1. Go to https://pypi.org/manage/account/publishing/
2. Add a new publisher:
   - **PyPI Project Name**: `llm-conductor`
   - **Owner**: `easel`
   - **Repository**: `llm-conductor`
   - **Workflow**: `release.yml`
   - **Environment**: `pypi`
3. Click **Add**

No token needed! GitHub Actions will authenticate automatically.

**Option B: API Token**

1. Generate token at https://pypi.org/manage/account/token/
2. Add to repository secrets as `PYPI_TOKEN`:
   - Go to repository **Settings** → **Secrets and variables** → **Actions**
   - Click **New repository secret**
   - Name: `PYPI_TOKEN`
   - Value: `pypi-...` (your token)
3. Uncomment the `password:` line in `.github/workflows/release.yml`:
   ```yaml
   - name: Publish to PyPI
     uses: pypa/gh-action-pypi-publish@release/v1
     with:
       password: ${{ secrets.PYPI_TOKEN }}  # Uncomment this line
   ```

### 3. Create PyPI Environments (Optional but Recommended)

1. Go to repository **Settings** → **Environments**
2. Click **New environment**
3. Name: `pypi`
4. Add protection rules (optional):
   - Required reviewers
   - Deployment branches: `main` only

## Creating a Release

### 1. Update Version and Changelog

The project uses dynamic versioning based on Git tags (`uv-dynamic-versioning`), so you don't need to manually update version numbers in code.

**Recommended:** Create/update `CHANGELOG.md` with release notes:

```markdown
## [1.0.0] - 2025-10-05

### Added
- New provider for Anthropic Claude
- Streaming support for all LiteLLM providers

### Fixed
- Batch processing error handling
- Token counting for multi-turn conversations

### Changed
- Improved retry logic with exponential backoff
```

### 2. Create and Push Tag

```bash
# Ensure you're on main branch and up to date
git checkout main
git pull origin main

# Create annotated tag (recommended)
git tag -a v1.0.0 -m "Release version 1.0.0"

# Or create lightweight tag
git tag v1.0.0

# Push tag to trigger release workflow
git push origin v1.0.0
```

**Version Format**: Must follow `v*.*.*` pattern (e.g., `v1.0.0`, `v0.2.1`, `v2.0.0-rc1`)

### 3. Monitor Release Workflow

1. Go to **Actions** tab in GitHub
2. Watch the "Release & Publish" workflow
3. The workflow will:
   - ✅ Build wheels and source distribution
   - ✅ Create GitHub Release with artifacts
   - ✅ Publish to PyPI
   - ✅ Deploy package index to GitHub Pages
   - ✅ Verify installations

### 4. Verify Release

After workflow completes (typically 3-5 minutes):

**Check GitHub Release:**
```bash
# Visit: https://github.com/easel/llm-conductor/releases/tag/v1.0.0
```

**Check PyPI:**
```bash
pip install llm-conductor==1.0.0
pip show llm-conductor
```

**Check GitHub Pages Index:**
```bash
pip install llm-conductor==1.0.0 --index-url https://easel.github.io/llm-conductor/simple/
```

## Pre-releases

For alpha/beta/release candidate versions:

```bash
git tag v1.0.0-rc1
git push origin v1.0.0-rc1
```

The workflow automatically marks releases as pre-release if the tag contains:
- `alpha`
- `beta`
- `rc`

## Rolling Back a Release

If you need to remove a bad release:

```bash
# Delete local tag
git tag -d v1.0.0

# Delete remote tag
git push origin :refs/tags/v1.0.0

# Delete GitHub Release (via web UI or gh CLI)
gh release delete v1.0.0

# Note: You cannot delete versions from PyPI
# Instead, use "yank" to mark them as unavailable:
# https://pypi.org/project/llm-conductor/
```

## Testing Releases

Before creating a production release, test with TestPyPI:

1. Get TestPyPI token from https://test.pypi.org/
2. Manually test the build:
   ```bash
   # Build package
   uv build

   # Upload to TestPyPI
   uv publish --index-url https://test.pypi.org/legacy/ --token pypi-...

   # Test installation
   pip install --index-url https://test.pypi.org/simple/ llm-conductor
   ```

## Troubleshooting

### Build Fails

**Issue**: `uv build` fails with version error

**Solution**: Ensure you're using annotated tags and have full git history:
```bash
git tag -d v1.0.0
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0 --force
```

### PyPI Publishing Fails

**Issue**: `403 Forbidden` error

**Solution**: Check trusted publishing configuration or PYPI_TOKEN secret

**Issue**: `400 Bad Request - File already exists`

**Solution**: You cannot overwrite PyPI releases. Increment version and create new tag.

### GitHub Pages Not Updating

**Issue**: Package index shows 404

**Solution**:
1. Check Pages settings are correct
2. Wait 2-3 minutes for deployment
3. Verify `gh-pages` branch was created
4. Check workflow logs for deployment errors

### Import Fails After Installation

**Issue**: `ModuleNotFoundError: No module named 'llm_conductor'`

**Solution**: Ensure Python 3.12 is being used:
```bash
python --version  # Should be 3.12.x
pip install llm-conductor --force-reinstall
```

## Release Checklist

- [ ] Update CHANGELOG.md with release notes
- [ ] Ensure all tests pass (`make test`)
- [ ] Ensure linting passes (`make lint`)
- [ ] Update documentation if needed
- [ ] Create and push version tag
- [ ] Monitor GitHub Actions workflow
- [ ] Verify GitHub Release was created
- [ ] Verify PyPI package is available
- [ ] Verify GitHub Pages index updated
- [ ] Test installation from both sources
- [ ] Announce release (optional)

## GitHub Pages Package Index

Users can install from the GitHub Pages index (useful for private deployments or PyPI downtime):

```bash
# Install latest version
pip install llm-conductor --index-url https://easel.github.io/llm-conductor/simple/

# Add to requirements.txt
--index-url https://easel.github.io/llm-conductor/simple/
llm-conductor==1.0.0

# Add to pyproject.toml
[[tool.uv.index]]
url = "https://easel.github.io/llm-conductor/simple/"
name = "llm-conductor-github"
```

## Automation Tips

### Pre-release Script

Create `.github/scripts/prepare-release.sh`:

```bash
#!/bin/bash
set -e

VERSION=$1
if [[ ! $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Usage: $0 <version> (e.g., 1.0.0)"
  exit 1
fi

# Run checks
make test
make lint

# Update changelog reminder
echo "📝 Don't forget to update CHANGELOG.md!"
echo ""
echo "Ready to release v$VERSION?"
read -p "Press enter to create and push tag..."

# Create and push tag
git tag -a "v$VERSION" -m "Release version $VERSION"
git push origin "v$VERSION"

echo "✅ Tag pushed! Monitor: https://github.com/easel/llm-conductor/actions"
```

Usage:
```bash
chmod +x .github/scripts/prepare-release.sh
./.github/scripts/prepare-release.sh 1.0.0
```
