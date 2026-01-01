# img2gs-local: Dokumentacja Techniczna (v1.0)

## 1. Przegląd Projektu
**img2gs-local** to aplikacja webowa typu "Single Image to 3D", działająca w pełni lokalnie. Służy do przekształcania pojedynczych zdjęć 2D w trójwymiarowe sceny przestrzenne, wykorzystując format Gaussian Splatting (.ply) do wizualizacji.

Aplikacja nie wymaga chmury ani zewnętrznych API, zapewniając pełną prywatność danych.

## 2. Architektura Systemu

### Stack Technologiczny
*   **Backend:** Python 3.10 + FastAPI (Asynchroniczna obsługa żądań).
*   **AI Model:** Depth Anything V2 (wersja Large) – State-of-the-Art w estymacjii głębokości jednoobrazowej.
*   **Frontend:** Vanilla JS + Three.js (Obsługa WebGL).
*   **Format Wyjściowy:** .PLY (zgodny ze standardem 3D Gaussian Splatting).

## 3. Algorytm Działania (Pipeline)

Proces konwersji jest **deterministyczny** (oparty na projekcji), a nie generatywny (oparty na halucynacji). Składa się z 5 kroków:

### Krok 1: Pre-processing (Super Resolution) **[Kluczowe dla jakości]**
Obraz wejściowy jest sztucznie powiększany (**Upscaling 2x** algorytmem Bicubic).
*   *Cel:* Zwiększenie liczby punktów 4-krotnie. Dzięki temu "luki" między pikselami w przestrzeni 3D stają się mikroskopijne, tworząc wrażenie litej powierzchni.

### Krok 2: Estymacja Głębokości (Depth Estimation)
Model AI (`Depth-Anything-V2-Large`) analizuje zdjęcie i generuje mapę głębi (obraz w skali szarości, gdzie jasność = odległość).

### Krok 3: Odszumianie (Denoising)
Na mapę głębi nakładany jest **Filtr Bilateralny** (OpenCV).
*   *Działanie:* Wygładza płaskie powierzchnie (ściany, podłogi) usuwając "falowanie" charakterystyczne dla modeli AI, jednocześnie **zachowując ostre krawędzie** obiektów.

### Krok 4: Projekcja 3D (Back-Projection)
Każdy piksel obrazu jest rzutowany w przestrzeń 3D wg wzoru:
*   `Z` (głębokość) = wzięte z mapy głębi.
*   `X, Y` = wyliczone z pozycji piksela i wirtualnego kąta widzenia kamery (FOV 55°).

### Krok 5: Generowanie Splatów
Zamiast zwykłych punktów, generujemy "Gaussian Splaty" – czyli zorientowane w przestrzeni elipsoidy/dyski.
*   W obecnej wersji "Simple Projection" splaty są ustawione **równolegle do kamery**.

## 4. Rendering (Viewer)

Wizualizacja odbywa się w przeglądarce przy użyciu silnika Three.js z następującymi optymalizacjami:

*   **Tryb "Opaque Cutout":** Wyłączone sortowanie przezroczystości (`transparent: false`, `alphaTest: 0.1`). Zapobiega to migotaniu (flickering) przy obracaniu modelu i drastycznie zwiększa wydajność.
*   **ACES Filmic Tone Mapping:** Kinowy standard mapowania kolorów, zapewniający realistyczny kontrast i brak "przepaleń".
*   **Czerń Absolutna:** Tło ustawione na `#050505` dla maksymalnego kontrastu (OLED-friendly).
*   **Crash Protection:** Automatyczne ograniczenie `pixelRatio` do 2.0 na ekranach 4K/Retina, aby zapobiec awariom pamięci VRAM.

## 5. Ograniczenia i Różnice vs "Pełny 3DGS"

| Cecha | img2gs-local (Obecny) | Pełny 3DGS (np. LGM / TripoSR) |
| :--- | :--- | :--- |
| **Metoda** | Projekcja (Matematyka) | Optymalizacja/Generacja (AI) |
| **Czas** | < 1 sekunda | 10-60 sekund |
| **Hardware** | Działa na CPU | Wymaga mocnego GPU (NVIDIA) |
| **Prawda** | Wierny zdjęciu (nie zmyśla) | Może halucynować nieistniejące obiekty |
| **Refleksy** | Brak (statyczny kolor) | Tak (Spherical Harmonics) |
| **Niewidoczne** | Czarne dziury za obiektami | Próba "domyślania się" tyłu |

## 6. Wymagania Sprzętowe
*   **RAM:** min. 4GB (Model zajmuje ok. 1.5GB).
*   **GPU:** Opcjonalne (przyspiesza estymację głębi, ale CPU jest wystarczająco szybkie).
*   **Dysk:** ok. 2GB na środowisko Python i wagi modelu.
