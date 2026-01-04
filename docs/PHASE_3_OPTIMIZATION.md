# Dokumentacja Modułu: 3D Gaussian Splatting Optimization

Ten dokument opisuje moduł optymalizacji 3D Gaussian Splatting (3DGS) zaimplementowany w ramach Fazy 3 projektu `img2gs`. Moduł ten odpowiada za przekształcenie wstępnej chmury punktów (wygenerowanej z obrazu 2D) w pełnoprawną reprezentację 3DGS, którą można renderować w czasie rzeczywistym.

## 1. Przegląd Architektury

Moduł opiera się na bibliotece `gsplat` (wersja 1.5.2), która dostarcza akcelerowaną przez CUDA implementację rasteryzacji różniczkowalnej.

### Kluczowe Komponenty

1.  **`core/optimization.py`**:
    *   **`GaussianOptimizer`**: Główna klasa zarządzająca procesem uczenia.
    *   **Inicjalizacja**: Konwertuje punkty 3D na parametry Gaussa (pozycje, log-skale, kwaterniony, logity przezroczystości, współczynniki Spherical Harmonics dla koloru).
    *   **Render**: Wywołuje `gsplat.rasterization` w celu wygenerowania obrazu z obecnych parametrów.
    *   **Optimize Step**: Oblicza stratę (L1 Loss) między wyrenderowanym obrazem a oryginałem i aktualizuje parametry metodą Adam.
    *   **Save PLY**: Eksportuje wynik do binarnego formatu `.ply` zgodnego ze standardowymi przeglądarkami 3DGS (np. Antimatter15, SuperSplat).

2.  **`run_optimization.py`**:
    *   Skrypt wejściowy (CLI) do uruchamiania optymalizacji na gotowych plikach.
    *   Obsługuje ładowanie chmur punktów, obrazów referencyjnych i zapis wyników.

3.  **`demo_pipeline.py`**:
    *   Skrypt "End-to-End".
    *   Łączy Fazę 2 (Depth Estimation + Projekcja) i Fazę 3 (Optymalizacja).
    *   Pozwala przejść od obrazka JPG/PNG prosto do modelu PLY.

## 2. Wymagania Środowiskowe

Ze względu na specyficzne wymagania biblioteki `gsplat`, środowisko musi być precyzyjnie skonfigurowane:

*   **System**: Windows 10/11 (lub Linux)
*   **GPU**: NVIDIA z obsługą CUDA
*   **Python**: 3.10
*   **PyTorch**: 2.4.1 (Wersja 2.5 nie jest jeszcze wspierana przez prekompilowane paczki gsplat na Windows)
*   **CUDA**: 12.1
*   **gsplat**: 1.5.2 (zainstalowane z dedykowanego index-url)

### Instrukcja Instalacji (Conda)

```bash
conda create -n img2gs-local python=3.10 -y
conda activate img2gs-local

# 1. PyTorch 2.4.1 dla CUDA 12.1
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# 2. Inne zależności
pip install numpy pillow plyfile

# 3. gsplat 1.5.2 (Wersja prekompilowana)
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu121
```

## 3. Instrukcja Użycia

### A. Uruchomienie Demo (Pełny automat)

Najprostszy sposób na przetestowanie działania. Skrypt bierze obrazek, tworzy z niego chmurę punktów i od razu ją optymalizuje.

```bash
python demo_pipeline.py nazwa_obrazka.jpg --iters 200
```

*   `--iters`: Liczba iteracji (domyślnie 100, zalecane 150-300 dla lepszej jakości).
*   **Wynik**: Plik `demo_output/optimized_gs.ply` oraz podgląd `render_result.png`.

### B. Ręczna Optymalizacja

Jeśli masz już wygenerowany plik `.ply` (z Fazy 2) i obrazek referencyjny:

```bash
python run_optimization.py --input "sciezka/do/chmury.ply" --image "sciezka/do/obrazka.png" --output "wynik.ply" --iterations 200
```

## 4. Wyjaśnienie Parametrów 3DGS

Każdy "Splat" w pliku wynikowym PLY posiada następujące atrybuty:

1.  **Position (x, y, z)**: Środek elipsoidy.
2.  **Scale (scale_0, scale_1, scale_2)**: Logarytm skali w trzech wymiarach.
3.  **Rotation (rot_0, rot_1, rot_2, rot_3)**: Kwaternion rotacji (unormowany).
4.  **Opacity**: Logit przezroczystości (wartość przed sigmoidą).
5.  **Color (f_dc_0, f_dc_1, f_dc_2)**: Współczynniki Spherical Harmonics (stopnia 0) reprezentujące kolor bazowy.

## 5. Podgląd Wyników

Wygenerowane pliki `.ply` są w formacie binarnym zgodnym ze standardem 3D Gaussian Splatting.

**Polecane przeglądarki:**
1.  **Antimatter15 Web Viewer**: Działa lokalnie w przeglądarce. Wystarczy przeciągnąć plik PLY.
    *   Adres: `https://antimatter15.com/splat/`
2.  **SuperSplat**: Bardziej zaawansowany edytor/przeglądarka oparta na PlayCanvas.
    *   Adres: `https://playcanvas.com/super-splat`

*Uwaga: Lokalny podgląd w `static/index.html` (jeśli uruchomisz serwer FastAPI) wyświetla obecnie model jako uproszczoną chmurę punktów, a nie pełny splatting.*
