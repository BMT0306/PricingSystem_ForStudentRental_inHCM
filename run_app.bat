@echo off
title He Thong Ho Tro Sinh Vien Tim Tro TP.HCM
color 0B
echo ======================================================================
echo    HE THONG HO TRO SINH VIEN TIM TRO O TP.HCM (AI PRICE EVALUATOR)
echo ======================================================================
echo.
echo Dang khoi dong may chu va mo trinh duyet web, vui long doi 3-5 giay...
echo.

cd /d "%~dp0"

if not exist ".venv\Scripts\streamlit.exe" (
    echo [LOI] Khong tim thay moi truong ao .venv! Vui long kiem tra lai thu muc.
    pause
    exit /b
)

:: Khoi dong Streamlit va tu dong mo trinh duyet
start "" http://localhost:8501
.\.venv\Scripts\streamlit.exe run app\streamlit_app.py --server.port=8501

pause
