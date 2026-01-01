# Implementacja Fazy 3: Skok Technologiczny (Prawdziwe 3DGS)

Ten dokument opisuje proces przejścia z obecnej metody "Projekcji Punktów" na pełną "Optymalizację Gaussów" (True 3D Gaussian Splatting), wykorzystując moc karty NVIDIA RTX 4070.

## 1. Wymagania Wstępne

*   **Sprzęt:** NVIDIA GPU z obsługą CUDA (Twój RTX 4070 jest idealny).
*   **Software:**
    *   CUDA Toolkit (np. wersja 11.8 lub 12.x).
    *   Visual Studio C++ Build Tools (wymagane do kompilacji kerneli CUDA).

## 2. Biblioteki (Tech Stack)

Zamiast pisać rasteryzator od zera, użyjemy nowoczesnej biblioteki **`gsplat`** od twórców Nerfstudio. Jest to obecnie najszybsza i najłatwiejsza w użyciu implementacja 3DGS w Pythonie.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install gsplat
```

## 3. Architektura Nowego Pipeline'u

Obecnie proces kończy się na **kroku 5** (wygenerowanie punktów). Faza 3 dodaje **krok 6: Optymalizacja**.

### Algorytm (Micro-Training Loop)

1.  **Inicjalizacja (Warm Start):**
    *   Bierzemy punkty wygenerowane przez `Depth Anything V2` (tak jak teraz).
    *   Zamiast traktować je jako wynik końcowy, traktujemy je jako **punkt startowy** dla optymalizatora.
    *   Każdy punkt dostaje parametry: `Pozycja (XYZ)`, `Rotacja (Quaternion)`, `Skala (3D)`, `Przezroczystość (Opacity)`, `Kolor (SH Coefficients)`.

2.  **Pętla Treningowa (Iteracje):**
    Wykonujemy np. 200 iteracji (co zajmie ok. 5-10 sekund na RTX 4070). W każdej iteracji:

    *   **Rasteryzacja (Forward):** Używamy `gsplat` do wyrenderowania obrazka z obecnych eliopsid.
    *   **Loss Calculation:** Porównujemy wyrenderowany obraz z oryginałem.
        *   *L1 Loss:* Czy kolory się zgadzają?
        *   *SSIM Loss:* Czy struktura obrazu jest podobna?
    *   **Wsteczna Propagacja (Backward):** Obliczamy gradienty – jak zmienić parametry każdego splata, żeby błąd zmalał.
    *   **Aktualizacja (Optimizer):** Zmieniamy kształt, pozycję i kolor splatów (używając Adama).
    *   **Zagęszczanie (Densification):** (Opcjonalnie co 50 iteracji) Dzielimy duże splaty na mniejsze, jeśli błąd w danym miejscu jest duży (np. na krawędziach obiektów).

3.  **Zapis Wyniku:**
    *   Zapisujemy finalne parametry do pliku `.ply`.

## 4. Dlaczego warto? (Efekty)

| Cecha | Obecnie (Projekcja) | Faza 3 (Optymalizacja) |
| :--- | :--- | :--- |
| **Powierzchnia** | Zbiór kropek (nawet przy Super Resolution) | **Ciągła tafla** (Gaussy się zlewają) |
| **Szczelność** | Widać dziury pod kątem | Gaussy się rozciągają (smeering), by łatać dziury |
| **Odbicia** | Brak (matowy kolor) | **Refleksy** (Spherical Harmonics) |
| **Detale** | Ograniczone rozdzielczością zdjęcia | Możliwość "sub-pikselowej" precyzji |

## 5. Przykładowy pseudo-kod (gsplat)

```python
import torch
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians

# 1. Inicjalizacja parametrów (z naszej chmury punktów)
means = torch.tensor(points_xyz, requires_grad=True, device="cuda")
scales = torch.tensor(initial_scales, requires_grad=True, device="cuda")
quats = torch.tensor(initial_rotations, requires_grad=True, device="cuda")
colors = torch.tensor(initial_colors, requires_grad=True, device="cuda")
opacities = torch.sigmoid(torch.tensor(initial_opacities, requires_grad=True, device="cuda"))

optimizer = torch.optim.Adam([means, scales, quats, colors, opacities], lr=0.001)

# 2. Pętla (Micro-Training)
for step in range(200):
    optimizer.zero_grad()
    
    # Render
    render_colors, _ = rasterize_gaussians(
        means, scales, quats, colors, opacities, 
        viewmatrix=view_mat, projmatrix=proj_mat
    )
    
    # Loss
    loss = torch.abs(render_colors - ground_truth_image).mean()
    
    # Update
    loss.backward()
    optimizer.step()

# 3. Save PLY...
```
