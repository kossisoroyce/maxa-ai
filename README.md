# Maxa AI: Eternal Inference with Theory of Mind

## Overview
Maxa is an AI assistant that maintains persistent memory and theory of mind capabilities, enabling more natural and context-aware interactions over time.

## Features
- **Persistent Memory**: Maintains context across conversations
- **Theory of Mind**: Understands user preferences and mental states
- **Eternal Inference**: Continuously learns and adapts to user needs
- **Modular Architecture**: Easy to extend and customize

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/maxa-ai.git
   cd maxa-ai
   ```

2. **Set up environment variables**
   Copy `.env.example` to `.env` and update with your API keys:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   uvicorn app.main:app --reload
   ```

5. **Access the API**
   - API docs: http://localhost:8000/docs
   - Redoc: http://localhost:8000/redoc

## Docker Setup

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Access services**
   - API: http://localhost:8000
   - Qdrant UI: http://localhost:6333
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (admin/admin)

## API Documentation

### Authentication
All endpoints require authentication using JWT tokens.

### Endpoints

#### Chat
- `POST /api/v1/chat` - Send a message and get a response
- `GET /api/v1/chat/ws/{conversation_id}` - WebSocket endpoint for real-time chat

#### Memory
- `POST /api/v1/memory` - Store a memory
- `GET /api/v1/memory` - Search memories
- `DELETE /api/v1/memory/{memory_id}` - Delete a memory

## Development

### Code Style
We use `black` for code formatting and `isort` for import sorting.

```bash
# Format code
black .

# Sort imports
isort .
```

### Testing
Run tests with pytest:

```bash
pytest tests/
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you use Maxa in your research, please cite:

```bibtex
@software{maxa2023,
  author = Kossiso Royce,
  title = {Maxa: Eternal Inference AI with Theory of Mind},
  year = {2025},
}
