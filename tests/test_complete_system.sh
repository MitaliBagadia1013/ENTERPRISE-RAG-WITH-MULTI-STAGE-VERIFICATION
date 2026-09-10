#!/bin/bash

echo ""
echo "VERIRAG - COMPLETE SYSTEM TEST"
echo ""
echo ""

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

TESTS_PASSED=0
TESTS_FAILED=0

run_test() {
    local test_name=$1
    local test_command=$2
    
    echo ""
    echo "Test: $test_name"
    echo ""
    
    if eval "$test_command"; then
        echo -e "${GREEN} PASSED${NC}: $test_name"
        ((TESTS_PASSED++))
    else
        echo -e "${RED} FAILED${NC}: $test_name"
        ((TESTS_FAILED++))
    fi
    echo ""
}

echo "1 Checking Redis..."
run_test "Redis Connection" "redis-cli ping > /dev/null 2>&1"

echo "2 Checking Environment Variables..."
run_test "OpenAI API Key" "grep -q 'OPENAI_API_KEY=sk-' .env"
run_test "Pinecone API Key" "grep -q 'PINECONE_API_KEY=' .env"
run_test "Cohere API Key" "grep -q 'COHERE_API_KEY=' .env"

echo "3 Testing Python Modules..."
run_test "Semantic Retrieval" "python3 test_semantic_retriever.py > /dev/null 2>&1"
run_test "Reranker" "python3 test_reranker.py > /dev/null 2>&1"
run_test "Self-Correction" "python3 test_self_correction.py > /dev/null 2>&1"
run_test "Semantic Cache" "python3 test_semantic_cache.py > /dev/null 2>&1"

echo "4 Testing FastAPI Server..."
echo "   Starting API server in background..."
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 > /tmp/api.log 2>&1 &
API_PID=$!
sleep 5

run_test "API Health Check" "curl -s http://localhost:8000/api/health | grep -q 'redis'"
run_test "API Root Endpoint" "curl -s http://localhost:8000/ | grep -q 'VeriRAG'"

echo "   Testing authentication..."
TOKEN=$(curl -s -X POST "http://localhost:8000/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin@company.com", "password": "admin123"}' | \
  python3 -c "import sys, json; print(json.load(sys.stdin).get('access_token', ''))" 2>/dev/null)

if [ ! -z "$TOKEN" ]; then
    echo -e "${GREEN} PASSED${NC}: JWT Authentication"
    ((TESTS_PASSED++))
    
    run_test "Cache Metrics Endpoint" "curl -s -H 'Authorization: Bearer $TOKEN' http://localhost:8000/api/cache/metrics | grep -q 'cache_hits'"
else
    echo -e "${RED} FAILED${NC}: JWT Authentication"
    ((TESTS_FAILED++))
fi

kill $API_PID 2>/dev/null

echo ""
echo ""
echo "TEST RESULTS"
echo ""
echo -e "${GREEN} Passed: $TESTS_PASSED${NC}"
echo -e "${RED} Failed: $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN} ALL TESTS PASSED! Your RAG system is ready!${NC}"
    exit 0
else
    echo -e "${YELLOW} Some tests failed. Check the output above.${NC}"
    exit 1
fi
