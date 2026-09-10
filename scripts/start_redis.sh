#!/bin/bash

set -e

echo "=========================================="
echo "  VeriRAG - Redis Setup"
echo "=========================================="

if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker Desktop."
    exit 1
fi

echo "Docker is running"

echo ""
echo "Cleaning up old containers..."
docker-compose down 2>/dev/null || true

echo ""
echo "Starting Redis containers..."
docker-compose up -d

echo ""
echo "Waiting for Redis to be ready..."
sleep 5

echo ""
echo "Running health checks..."
if docker exec verirag-redis redis-cli ping | grep -q PONG; then
    echo "Redis is healthy and responding"
else
    echo "Redis health check failed"
    exit 1
fi

echo ""
echo "Container Status:"
docker-compose ps

echo ""
echo "Redis Info:"
docker exec verirag-redis redis-cli INFO server | grep -E "redis_version|os|arch|process_id"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Redis Server:"
echo "  Host: localhost"
echo "  Port: 6379"
echo "  Status: Running"
echo ""
echo "Redis Insight (Web UI):"
echo "  URL: http://localhost:8001"
echo "  Status: Running"
echo ""
echo "Useful Commands:"
echo "  Stop:    docker-compose down"
echo "  Logs:    docker-compose logs -f redis"
echo "  Monitor: docker exec -it verirag-redis redis-cli MONITOR"
echo "  Flush:   docker exec -it verirag-redis redis-cli FLUSHALL"
echo ""
echo "To test the cache, run:"
echo "  python test_semantic_cache.py"
echo ""
