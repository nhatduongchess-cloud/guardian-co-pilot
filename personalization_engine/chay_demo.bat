@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Guardian - Vertical 1 (Personalization)

where python >nul 2>nul
if errorlevel 1 (
  echo [Loi] Khong tim thay Python. Cai Python 3.11+ roi thu lai.
  echo.
  pause
  exit /b 1
)

:menu
cls
echo ==================================================
echo    GUARDIAN  -  VERTICAL 1   Personalization Engine
echo ==================================================
echo.
echo   [1] Demo ca nhan hoa          (simulate_trips)   ^<-- mac dinh
echo   [2] Demo 10 chuyen THAT        (run_real_data)
echo   [3] So lieu bang chung         (generate_evidence)
echo   [4] Chay toan bo test          (pytest)
echo   [5] Mo trang tong hop ket qua  (HTML)
echo   [6] Phan tich DATASET that -^> video  (analyze_dataset)
echo   [7] Cua so LIVE de record            (analyze --show, Q de dung)
echo   [0] Thoat
echo.
set "c=1"
set /p "c=Chon [1-7], Enter = 1: "

if "%c%"=="1" goto d_sim
if "%c%"=="2" goto d_real
if "%c%"=="3" goto d_evi
if "%c%"=="4" goto d_test
if "%c%"=="5" goto d_html
if "%c%"=="6" goto d_anavid
if "%c%"=="7" goto d_analive
if "%c%"=="0" exit /b 0
goto menu

:d_sim
echo.
python -m mock_data.simulate_trips
goto done

:d_real
echo.
if not exist "evidence\driver_state_realdata.json" (
  echo [Buoc 1/2] Chay perception tren footage that (can dataset zip trong Downloads)...
  python -m evidence.perceive_driver
)
echo [Chay] Vertical 1 tren 10 chuyen that...
python -m evidence.run_real_data
goto done

:d_evi
echo.
python -m evidence.generate_evidence
goto done

:d_test
echo.
python -m pytest -q
goto done

:d_html
start "" "Guardian-Vertical1-Personalization.html"
goto menu

:d_anavid
echo.
echo [Chay] Phan tich tren dataset that cua BGK (can vai phut, can dataset zip trong Downloads)...
python -m evidence.analyze_dataset
if exist "demo\vertical1_dataset_analysis.mp4" start "" "demo\vertical1_dataset_analysis.mp4"
goto done

:d_analive
echo.
echo [LIVE] Cua so phan tich se mo. Bat phan mem quay man hinh, roi xem.
echo        Nhan Q trong cua so video de dung.
python -m evidence.analyze_dataset --show
goto done

:done
echo.
echo --------------------------------------------------
echo Xong. Nhan phim bat ky de quay lai menu...
pause >nul
goto menu
