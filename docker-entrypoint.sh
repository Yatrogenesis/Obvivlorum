#!/bin/bash
set -e

# Obvivlorum Docker entrypoint script
echo "Starting OBVIVLORUM AI Symbiote..."
echo "User: $(whoami)"
echo "Working directory: $(pwd)"

# Initialize environment
if [ ! -f .obvivlorum_initialized ]; then
    echo "First time setup..."
    python -c "print('OBVIVLORUM AI - Docker environment ready')"
    touch .obvivlorum_initialized
fi

# Execute passed command
exec "$@"