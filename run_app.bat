@echo off
chcp 65001 >nul
title HỆ THỐNG ĐỊNH GIÁ & TÌM PHÒNG TRỌ SINH VIÊN TP.HCM
color 0B
cd /d "%~dp0"

echo ==============================================================================
echo    HỆ THỐNG AI ĐỊNH GIÁ & HỖ TRỢ TÌM PHÒNG TRỌ SINH VIÊN TP.HCM
echo ==============================================================================
echo.
echo [*] Đang khởi động máy chủ Streamlit...
echo [*] Trình duyệt web sẽ tự động bật lên ngay khi hệ thống sẵn sàng.
echo [*] Nhấn Ctrl+C hoặc đóng cửa sổ này khi muốn dừng web.
echo.

if not exist ".venv\Scripts\streamlit.exe" (
    echo [LỖI] Không tìm thấy môi trường ảo .venv!
    echo Vui lòng kiểm tra lại thư mục hoặc cài đặt thư viện trước khi chạy.
    pause
    exit /b
)

:: Chạy Streamlit (Streamlit mặc định sẽ tự bật trình duyệt khi server sẵn sàng)
.\.venv\Scripts\streamlit.exe run app\streamlit_app.py --server.port=8501 --server.address=localhost

pause
