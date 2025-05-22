# Contributing to Maxa AI

We're excited that you're interested in contributing to Maxa AI! This document outlines the process for contributing to the project.

## Code of Conduct

This project adheres to the Contributor Covenant [code of conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
   ```bash
   git clone https://github.com/your-username/maxa-ai.git
   cd maxa-ai
   ```
3. **Set up the development environment**
   ```bash
   make install-dev
   ```
4. **Create a branch** for your changes
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

1. **Make your changes**
2. **Run tests** to ensure everything works
   ```bash
   make test
   ```
3. **Format and lint** your code
   ```bash
   make format
   make lint
   ```
4. **Commit your changes** with a descriptive message
   ```bash
   git commit -m "Add your commit message here"
   ```
5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Create a Pull Request** from your fork to the main repository

## Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code
- Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Keep lines under 88 characters (Black's default)
- Type hints are required for all function signatures

## Testing

- Write tests for all new features and bug fixes
- Ensure all tests pass before submitting a PR
- Use descriptive test function names (e.g., `test_function_name_should_do_x_when_y`)

## Documentation

- Update documentation when adding new features or changing behavior
- Keep docstrings up to date
- Add examples for complex functionality

## Pull Request Process

1. Ensure your code passes all tests and linters
2. Update the documentation as needed
3. Ensure the test coverage remains high
4. Request review from at least one maintainer
5. Address any feedback from code reviews

## Reporting Issues

When reporting issues, please include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs. actual behavior
- Version information (OS, Python version, package versions)
- Any relevant error messages or logs

## Feature Requests

For feature requests, please:

1. Check if the feature has already been requested
2. Explain why this feature would be valuable
3. Include any relevant use cases or examples

## License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
