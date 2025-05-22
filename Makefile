.PHONY: help install install-dev format lint test test-cov clean run docker-build docker-up docker-down docker-logs

# Help
help:
	@echo "Available commands:"
	@echo "  make install         Install production dependencies"
	@echo "  make install-dev     Install development dependencies"
	@echo "  make format          Format code with black and isort"
	@echo "  make lint            Run linters (black, isort, flake8, mypy)"
	@echo "  make test            Run tests"
	@echo "  make test-cov        Run tests with coverage"
	@echo "  make clean           Clean up temporary files"
	@echo "  make run             Run the application"
	@echo "  make docker-build    Build Docker images"
	@echo "  make docker-up       Start services with Docker Compose"
	@echo "  make docker-down     Stop and remove containers"
	@echo "  make docker-logs     View container logs"

# Installation
install:
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-dev.txt
	pre-commit install

# Code formatting
format:
	black .
	isort .


# Linting
lint:
	black --check .
	isort --check-only .
	flake8 .
	mypy .


# Testing
test:
	pytest tests/

test-cov:
	pytest --cov=app --cov-report=term-missing tests/


# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +
	rm -rf .coverage htmlcov

# Run the application
run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Docker commands
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f
