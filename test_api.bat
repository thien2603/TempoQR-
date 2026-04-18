@echo off
echo Testing TempoQR FastAPI...
echo.

echo === Test 1: Health Check ===
curl -X GET http://localhost:8000/
echo.

echo === Test 2: Simple Question ===
curl -X POST http://localhost:8000/ask ^
-H "Content-Type: application/json" ^
-d "{\"question\": \"Who is the CEO of Apple?\"}"
echo.

echo === Test 3: Temporal Question ===
curl -X POST http://localhost:8000/ask ^
-H "Content-Type: application/json" ^
-d "{\"question\": \"Who was president of USA in 2020?\"}"
echo.

echo === Test 4: Complex Question ===
curl -X POST http://localhost:8000/ask ^
-H "Content-Type: application/json" ^
-d "{\"question\": \"What company did Steve Jobs found?\"}"
echo.

echo === Test 5: Invalid Question (Error Handling) ===
curl -X POST http://localhost:8000/ask ^
-H "Content-Type: application/json" ^
-d "{\"question\": \"\"}"
echo.

pause
