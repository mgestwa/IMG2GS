# Plan Rozwoju Projektu `img2gs-local`

Poniższy dokument przedstawia strategiczny plan rozwoju aplikacji, podzielony na fazy – od prostych ulepszeń po zmianę technologii bazowej.

## Faza 1: Udoskonalenie Geometrii (Modele AI)
**Cel:** Uzyskanie "metrycznej" poprawności i jeszcze większej szczegółowości detali.

1.  **Wdrożenie Marigold (Diffusion-based Scale-Invariant Depth)**
    *   **Opis:** Zamiana obecnego modelu `Depth Anything V2` na `Marigold`.
    *   **Dlaczego:** Marigold generuje głębię o ostrości zbliżonej do samego zdjęcia. Widoczne są drobne faktuły (cegiełki, słoje drewna), które `Depth Anything` wygładza.
    *   **Koszt:** Czas generowania wzrośnie (z ~0.5s do ~3-5s na GPU), ale jakość "statyczna" będzie topowa.
2.  **ZoeDepth (Metryczna głębia)**
    *   **Opis:** Model trenowany na rzeczywistych pomiarach.
    *   **Zastosowanie:** Pozwoliłoby to na dodanie "linijki" w aplikacji – użytkownik mógłby zmierzyć odległość w metrach.

## Faza 2: Funkcjonalność i Eksport
**Cel:** Uczynienie narzędzia użytecznym w pracy (Designerzy, Architekci).

1.  **Eksport do Siatki (.OBJ / .GLB)**
    *   **Problem:** Obecnie mamy chmurę punktów (.PLY). Programy CAD/3D wolą siatki (trójkąty).
    *   **Rozwiązanie:** Implementacja algorytmu *Poisson Surface Reconstruction* lub *Ball Pivoting* (biblioteka Open3D) po stronie backendu.
    *   **Efekt:** Użytkownik pobiera plik, który może wrzucić prosto do Blendera/SektchUp i nałożyć teksturę.
2.  **Galeria Lokalna**
    *   Dodanie prostego paska bocznego z historią wygenerowanych modeli, aby można było do nich wracać bez ponownego uploadu.

## Faza 3: Skok Technologiczny – "Prawdziwe" 3DGS (REKOMENDOWANA DLA RTX 4070)
**Status Hardware:** Twoja karta **NVIDIA RTX 4070** jest idealna do tego zadania (dużo VRAM, rdzenie Tensor).

**Cel:** Przejście z "Projekcji Punktów" na "Optymalizację Gaussów".

*   **Technologia:** Wykorzystanie biblioteki `gsplat` (od Nerfstudio) lub `diff-gaussian-rasterization`.
*   **Obecnie:** Rzutujemy piksel 2D -> punkt 3D. To jest szybkie (<1s), ale statyczne.
*   **Plan:**
    1.  Instalacja biblioteki CUDA (`pip install gsplat`).
    2.  Implementacja pętli treningowej (ok. 100-200 iteracji na scenę).
    3.  Optymalizacja koloru, przezroczystości i **obrotu** każdego splata.
*   **Efekt:**
    *   Trening potrwa ok. 10-15 sekund na RTX 4070.
    *   Uzyskamy modele z refleksami (Shperical Harmonics) i idealną ciągłością powierzchni.

## Faza 4: Skos na Przeglądarkę (WebGPU)
**Cel:** Wyeliminowanie backendu Pythonowego dla użytkowników końcowych.

1.  **Inferencja w JS (ONNX Runtime Web)**
    *   Przekonwertowanie modelu Depth-Anything do formatu ONNX.
    *   Uruchomienie go bezpośrednio w przeglądarce klienta.
2.  **Korzyści:** Aplikację można wrzucić na GitHub Pages (hosting statyczny). Działa offline, zero kosztów serwera, pełna prywatność (zdjęcia nie opuszczają RAMu przeglądarki).

## Rekomendacja "Co dalej?"
Jeśli masz mocną kartę graficzną (NVIDIA), polecam zacząć od **Fazy 3** (Prawdziwe 3DGS), bo to daje największy efekt "WOW".
Jeśli zależy Ci na użyteczności inżynierskiej, wybierz **Fazę 2** (Eksport do Mesh).
