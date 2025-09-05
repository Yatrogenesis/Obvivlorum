# Contributing to Obvivlorum AI Symbiote System

We're thrilled that you're interested in contributing to the Obvivlorum project! This document provides guidelines and information about how you can contribute effectively.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contributing Guidelines](#contributing-guidelines)
5. [Pull Request Process](#pull-request-process)
6. [Coding Standards](#coding-standards)
7. [Testing Guidelines](#testing-guidelines)
8. [Documentation](#documentation)
9. [Research Contributions](#research-contributions)
10. [Community](#community)

## Code of Conduct

This project and everyone participating in it are governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [pako.molina@gmail.com](mailto:pako.molina@gmail.com).

## Getting Started

### Types of Contributions

We welcome various types of contributions:

- **Code Contributions**: Bug fixes, new features, performance improvements
- **Research Contributions**: Theoretical frameworks, consciousness metrics, quantum processing
- **Documentation**: Improving existing docs, adding examples, creating tutorials
- **Testing**: Writing tests, reporting bugs, improving test coverage
- **Design**: UI/UX improvements, system architecture proposals
- **Community**: Helping other users, improving processes, organizing events

### Before You Start

1. **Search existing issues** to see if your contribution is already being discussed
2. **Create an issue** to discuss major changes before implementation
3. **Fork the repository** and create a feature branch
4. **Read the documentation** to understand the system architecture

## Development Setup

### Prerequisites

- Python 3.11+ 
- Git
- Docker and Docker Compose (for sandbox testing)
- WSL2 (for Windows users wanting Linux integration)

### Installation

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/your-username/Obvivlorum.git
   cd Obvivlorum
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv obvivlorum-env
   source obvivlorum-env/bin/activate  # On Windows: obvivlorum-env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Run initial tests:**
   ```bash
   python test_system_final.py
   ```

5. **Verify launcher works:**
   ```bash
   python unified_launcher.py --list
   ```

### Development Dependencies

For full development capabilities, install additional packages:

```bash
pip install pytest pytest-asyncio pytest-cov black flake8 mypy jupyter numba qiskit pennylane
```

## Contributing Guidelines

### Issue Guidelines

- **Use issue templates** when available
- **Provide clear, descriptive titles**
- **Include reproduction steps** for bugs
- **Add relevant labels** and assign to appropriate milestone
- **Reference related issues** using `#issue-number`

### Branch Naming

Use descriptive branch names with prefixes:

- `feature/quantum-processing-enhancement`
- `bugfix/security-manager-jwt-issue`
- `docs/api-reference-update` 
- `research/consciousness-metrics-validation`
- `refactor/core-orchestrator-cleanup`

### Commit Messages

Follow conventional commit format:

```
<type>(<scope>): <description>

<body>

<footer>
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `research`

**Examples:**
```bash
feat(security): add JWT token expiration handling

Implements automatic token refresh and expiration validation
to improve system security and user experience.

Closes #123
```

```bash
research(consciousness): implement IIT phi calculation

Adds rigorous Integrated Information Theory implementation
with mathematical validation and performance optimization.

Co-authored-by: Dr. Research Partner <research@university.edu>
```

## Pull Request Process

### Before Submitting

1. **Update your branch** with the latest main
2. **Run all tests** and ensure they pass
3. **Run code formatting** and linting
4. **Update documentation** if needed
5. **Add/update tests** for your changes

### Pull Request Template

When creating a PR, use this structure:

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Research contribution

## Testing
- [ ] New tests added
- [ ] All tests pass
- [ ] Manual testing completed

## Research Impact (if applicable)
- [ ] Advances consciousness research capabilities
- [ ] Improves theoretical framework implementation
- [ ] Adds new scientific metrics or validation

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] New and existing unit tests pass locally with my changes
```

### Review Process

1. **Automated checks** must pass (CI/CD pipeline)
2. **At least one maintainer review** required
3. **Research contributions** may require additional expert review
4. **Documentation updates** should be reviewed for accuracy
5. **Breaking changes** require special attention and discussion

## Coding Standards

### Python Style

We follow PEP 8 with these specifics:

- **Line length**: 88 characters (Black formatter default)
- **Import organization**: Use `isort` for consistent import ordering
- **Type hints**: Required for public APIs and recommended everywhere
- **Docstrings**: Google-style docstrings for all public functions/classes

### Code Quality Tools

Run these before submitting:

```bash
# Format code
black .
isort .

# Lint code
flake8 .
pylint obvivlorum/  # If you create an obvivlorum package

# Type checking
mypy .

# Security check
bandit -r .
```

### Architecture Principles

- **Separation of Concerns**: Each module should have a single responsibility
- **Dependency Injection**: Use dependency injection for testability
- **Async/Await**: Use async patterns for I/O operations
- **Error Handling**: Comprehensive error handling with proper logging
- **Security First**: Never compromise on security features

### Research Code Standards

For research contributions:

- **Mathematical Rigor**: Include mathematical foundations and references
- **Reproducibility**: Code should be reproducible with consistent results
- **Documentation**: Extensive documentation of algorithms and theory
- **Validation**: Include validation against known results where possible
- **Performance**: Consider computational complexity and optimization

## Testing Guidelines

### Test Structure

```
tests/
├── unit/              # Unit tests for individual components
├── integration/       # Integration tests between components
├── research/          # Tests for research algorithms and metrics
├── performance/       # Performance benchmarks
└── fixtures/          # Test data and fixtures
```

### Writing Tests

- **Use pytest** as the testing framework
- **Follow AAA pattern**: Arrange, Act, Assert
- **Test edge cases** and error conditions
- **Use meaningful test names** that describe what's being tested
- **Mock external dependencies** to ensure isolated testing

### Example Test

```python
import pytest
from unittest.mock import Mock, AsyncMock
from security_manager import SecurityManager, PrivilegeLevel

class TestSecurityManager:
    @pytest.fixture
    def security_manager(self):
        return SecurityManager()
    
    @pytest.mark.asyncio
    async def test_privilege_escalation_requires_approval(self, security_manager):
        # Arrange
        initial_level = PrivilegeLevel.USER
        target_level = PrivilegeLevel.ADMIN
        
        # Act
        result = await security_manager.set_privilege_level(target_level)
        
        # Assert
        assert result == False  # Should require approval
        assert security_manager.current_privilege_level == initial_level
```

### Research Test Guidelines

- **Validate theoretical results** against published benchmarks
- **Test mathematical properties** (commutativity, associativity, etc.)
- **Performance benchmarks** with statistical significance
- **Reproducibility tests** with fixed random seeds

## Documentation

### Types of Documentation

- **API Documentation**: Automatically generated from docstrings
- **User Guides**: Step-by-step usage instructions
- **Architecture Docs**: System design and component interaction
- **Research Papers**: Theoretical foundations and validation
- **Examples**: Practical usage examples and tutorials

### Documentation Standards

- **Clear and Concise**: Write for your audience's expertise level
- **Code Examples**: Include working code examples
- **Keep Updated**: Update docs when making code changes
- **Link References**: Link to relevant papers, equations, and external resources

### Building Documentation

```bash
# Generate API documentation
pydoc-markdown

# Build research documentation
jupyter nbconvert --to html research_notebooks/*.ipynb

# Test documentation examples
python -m doctest docs/examples/*.md
```

## Research Contributions

### Theoretical Contributions

We especially welcome contributions in:

- **Consciousness Metrics**: IIT, GWT, and novel consciousness measures
- **Quantum Processing**: Quantum-inspired algorithms and simulation
- **Neuroplasticity**: Computational models of synaptic plasticity
- **Meta-Recursive Systems**: Self-improving AI architectures
- **Holographic Memory**: Information storage and retrieval models

### Research Standards

- **Literature Review**: Reference existing work and position your contribution
- **Mathematical Foundation**: Provide rigorous mathematical basis
- **Experimental Validation**: Include validation experiments when applicable
- **Reproducibility**: Ensure results can be reproduced by others
- **Peer Review**: Welcome peer review and incorporate feedback

### Publishing and Attribution

- **Co-authorship**: Research contributors may be included as co-authors
- **Attribution**: Proper citation of the Obvivlorum framework required
- **Open Science**: Results should be made publicly available
- **Collaboration**: We encourage collaborative research projects

## Community

### Getting Help

- **GitHub Discussions**: General questions and discussions
- **Issues**: Bug reports and feature requests  
- **Email**: For sensitive matters or research collaborations
- **Documentation**: Check existing docs first

### Community Guidelines

- **Be Respectful**: Treat all community members with respect
- **Be Constructive**: Provide constructive feedback and criticism
- **Be Patient**: Remember that this is an open source project
- **Be Inclusive**: Welcome contributors from all backgrounds

### Recognition

Contributors are recognized through:

- **Contributor List**: Listed in project documentation
- **Release Notes**: Mentioned in release announcements
- **Academic Papers**: Co-authorship on research publications
- **Conference Presentations**: Speaking opportunities at conferences

## Development Workflow

### Typical Contribution Workflow

1. **Identify** a bug, feature, or research opportunity
2. **Discuss** in an issue before major work
3. **Fork** and create a feature branch
4. **Develop** following coding standards
5. **Test** thoroughly with appropriate test coverage
6. **Document** your changes
7. **Submit** a pull request
8. **Respond** to review feedback
9. **Celebrate** your contribution!

### Release Process

Releases follow semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes or major architectural changes
- **MINOR**: New features, research contributions
- **PATCH**: Bug fixes, documentation improvements

### Maintenance

- **Active maintenance** of main branch
- **Security updates** prioritized
- **Long-term support** for major versions
- **Community involvement** in maintenance decisions

## Questions?

If you have questions about contributing, please:

1. **Search existing issues** and discussions
2. **Check the documentation**
3. **Create a new issue** with the question label
4. **Email the maintainers** for private matters

Thank you for contributing to Obvivlorum! Together, we're advancing the frontier of consciousness research and AI development.

---

*This contributing guide is maintained by the Obvivlorum community and is updated regularly to reflect best practices and community feedback.*