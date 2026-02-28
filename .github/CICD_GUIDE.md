# CI/CD Pipeline Guide

## Overview

This project uses GitHub Actions for continuous integration and deployment. The pipeline automatically tests, builds, and deploys the Quantum-AI Smart Energy Load Balancing System.

## Workflows

### 1. CI/CD Pipeline (`ci-cd.yml`)
**Triggers:** Push to main, T2-Quant, develop branches; Pull requests

**Jobs:**
- **Test Suite**: Runs on Python 3.11 and 3.12
  - Installs dependencies
  - Runs linting with flake8
  - Executes all unit and property-based tests
  - Generates coverage reports
  - Uploads to Codecov
  
- **Build**: Creates Docker image
  - Sets up Docker Buildx
  - Builds image with commit SHA tag
  - Tests the built image
  
- **Security**: Scans for vulnerabilities
  - Runs safety check on dependencies
  - Executes bandit security scan
  
- **Deploy Staging**: Deploys to staging (T2-Quant branch)
  - Automatic deployment on T2-Quant branch
  
- **Deploy Production**: Deploys to production (main branch)
  - Automatic deployment on main branch
  - Creates release tags

### 2. Fast Tests (`test-fast.yml`)
**Triggers:** Push to any branch; All pull requests

**Purpose:** Quick validation for rapid feedback

**Jobs:**
- Quick validation with core tests
- Import validation
- Runs in ~2 minutes

### 3. Docker Build & Publish (`docker-publish.yml`)
**Triggers:** Push to main/T2-Quant; Version tags; Releases

**Purpose:** Build and publish Docker images to GitHub Container Registry

**Features:**
- Multi-platform builds
- Automatic tagging (branch, version, SHA)
- Layer caching for faster builds
- Published to ghcr.io

### 4. Code Quality (`code-quality.yml`)
**Triggers:** Push to main, T2-Quant, develop; Pull requests

**Purpose:** Maintain code quality standards

**Checks:**
- Black formatting
- isort import sorting
- flake8 linting
- pylint analysis
- Cyclomatic complexity (radon)
- Maintainability index

## Status Badges

Add these badges to your README to show pipeline status:

```markdown
[![CI/CD Pipeline](https://github.com/HARSH83022/Quantum-AI-Smart-Energy-Load-Balancing-System-for-Smart-Cities/actions/workflows/ci-cd.yml/badge.svg?branch=T2-Quant)](https://github.com/HARSH83022/Quantum-AI-Smart-Energy-Load-Balancing-System-for-Smart-Cities/actions/workflows/ci-cd.yml)
[![Fast Tests](https://github.com/HARSH83022/Quantum-AI-Smart-Energy-Load-Balancing-System-for-Smart-Cities/actions/workflows/test-fast.yml/badge.svg)](https://github.com/HARSH83022/Quantum-AI-Smart-Energy-Load-Balancing-System-for-Smart-Cities/actions/workflows/test-fast.yml)
[![Code Quality](https://github.com/HARSH83022/Quantum-AI-Smart-Energy-Load-Balancing-System-for-Smart-Cities/actions/workflows/code-quality.yml/badge.svg)](https://github.com/HARSH83022/Quantum-AI-Smart-Energy-Load-Balancing-System-for-Smart-Cities/actions/workflows/code-quality.yml)
[![Docker Build](https://github.com/HARSH83022/Quantum-AI-Smart-Energy-Load-Balancing-System-for-Smart-Cities/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/HARSH83022/Quantum-AI-Smart-Energy-Load-Balancing-System-for-Smart-Cities/actions/workflows/docker-publish.yml)
```

## Environment Variables

The following secrets need to be configured in GitHub repository settings:

### Required for CI/CD
- `GITHUB_TOKEN` - Automatically provided by GitHub Actions

### Optional for Enhanced Features
- `CODECOV_TOKEN` - For coverage reporting
- `DOCKER_USERNAME` - For Docker Hub publishing
- `DOCKER_PASSWORD` - For Docker Hub authentication
- `RENDER_API_KEY` - For Render deployment
- `IBM_QUANTUM_API_KEY` - For quantum backend tests

## Branch Strategy

```
main (production)
  ↑
  └── T2-Quant (staging)
        ↑
        └── develop (development)
              ↑
              └── feature/* (features)
```

### Deployment Flow
1. **Feature branches** → Create PR to `develop`
2. **Develop** → Merge to `T2-Quant` for staging
3. **T2-Quant** → Merge to `main` for production

## Running Locally

To run the same checks locally before pushing:

```bash
# Install dev dependencies
pip install pytest pytest-cov flake8 black isort pylint bandit safety

# Run tests
pytest tests/ -v --cov=src

# Check formatting
black --check src tests
isort --check-only src tests

# Run linting
flake8 src tests --max-line-length=127

# Security scan
bandit -r src/
safety check

# Build Docker
docker build -t quantum-energy-system .
```

## Troubleshooting

### Tests Failing in CI but Pass Locally
- Check Python version (CI uses 3.11 and 3.12)
- Verify environment variables are set in GitHub Secrets
- Check for platform-specific issues (CI runs on Ubuntu)

### Docker Build Failing
- Ensure Dockerfile is up to date
- Check that all dependencies are in requirements.txt
- Verify base image is accessible

### Coverage Upload Failing
- Add `CODECOV_TOKEN` to GitHub Secrets
- Check Codecov integration is enabled

### Deployment Failing
- Verify deployment secrets are configured
- Check deployment platform status
- Review deployment logs in Actions tab

## Monitoring

### View Pipeline Status
1. Go to repository on GitHub
2. Click "Actions" tab
3. Select workflow to view runs
4. Click on specific run for detailed logs

### Notifications
- GitHub sends email notifications for failed workflows
- Configure Slack/Discord webhooks for team notifications
- Set up status checks for pull requests

## Best Practices

1. **Always run tests locally** before pushing
2. **Keep workflows fast** - Use caching and parallel jobs
3. **Fail fast** - Run quick tests first
4. **Secure secrets** - Never commit credentials
5. **Monitor coverage** - Aim for >80% code coverage
6. **Review security scans** - Address vulnerabilities promptly
7. **Tag releases** - Use semantic versioning (v1.0.0)

## Performance Metrics

Current pipeline performance:
- **Fast Tests**: ~2 minutes
- **Full CI/CD**: ~8-10 minutes
- **Docker Build**: ~5 minutes
- **Code Quality**: ~3 minutes

## Future Enhancements

- [ ] Add integration tests with real database
- [ ] Implement E2E API tests
- [ ] Add performance benchmarking
- [ ] Set up automated dependency updates (Dependabot)
- [ ] Add load testing in staging
- [ ] Implement blue-green deployment
- [ ] Add rollback automation

## Support

For CI/CD issues:
1. Check workflow logs in Actions tab
2. Review this guide
3. Check GitHub Actions documentation
4. Open an issue with workflow run link

---

**Last Updated**: 2026-03-01
**Maintained By**: Development Team
