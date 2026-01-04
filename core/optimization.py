import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from gsplat import rasterization
import os
import plyfile

class GaussianOptimizer:
    """
    EDUCATIONAL: Klasa zarządzająca pełnym procesem optymalizacji 3D Gaussian Splattingu.
    
    W klasycznym renderingu mamy siatkę trójkątów. W 3DGS mamy miliony "gaussów" (elipsoid 3D).
    Każdy gauss ma:
    1. Pozycję (XYZ) - gdzie jest w przestrzeni.
    2. Skalę (3 wartości) - jak bardzo jest rozciągnięty w 3 osiach.
    3. Rotację (Kwaternion) - jak jest obrócony.
    4. Kolor (Spherical Harmonics) - jak wygląda zależnie od kąta patrzenia.
    5. Przezroczystość (Opacity) - jak bardzo zasłania to co za nim.
    
    Proces uczenia to "odwrócenie" procesu renderowania.
    1. Forward: Renderujemy obraz z obecnych parametrów.
    2. Loss: Liczymy błąd względem zdjęcia referencyjnego.
    3. Backward: Liczymy gradienty (jak zmienić parametry, by zmniejszyć błąd).
    4. Step: Aktualizujemy parametry.
    """
    def __init__(self, points: np.ndarray, colors: np.ndarray, device="cuda"):
        """
        Inicjalizacja optymalizatora punktami startowymi.
        
        Args:
            points: [N, 3] Pozycje początkowe (np. z Depth Anything).
            colors: [N, 3] Kolory początkowe (RGB, 0-1).
            device: 'cuda' jest wymagana dla gsplat.
        """
        self.device = device
        # Ensure input is float32
        self.points_initial = torch.tensor(points, dtype=torch.float32, device=device)
        self.colors_initial = torch.tensor(colors, dtype=torch.float32, device=device)
        self.N = self.points_initial.shape[0]
        
        print(f"[GaussianOptimizer] Inicjalizacja dla {self.N} punktów na urządzeniu {device}.")
        self._initialize_parameters()
        self._setup_optimizer()

    def _initialize_parameters(self):
        """
        Tworzy trenowalne parametry (torch.nn.Parameter).
        To właśnie te zmienne będą modyfikowane przez algorytm gradientowy.
        """
        # 1. Means (Pozycje): Startujemy dokładnie tam, gdzie są punkty.
        self.means = nn.Parameter(self.points_initial.clone())
        
        # 2. Scales (Skale): Startujemy z małymi kulkami.
        # EDUCATIONAL: Używamy log-space, żeby skale były zawsze dodatnie po exp().
        # -5.0 to po exp() ok. 0.006 jednostki.
        self.scales = nn.Parameter(torch.full((self.N, 3), -5.0, device=self.device))
        
        # 3. Rotations (Kwaterniony): Startujemy z tożsamością (brak obrotu).
        # [1, 0, 0, 0] to kwaternion jednostkowy (w, x, y, z).
        quats = torch.zeros((self.N, 4), device=self.device)
        quats[:, 0] = 1.0
        self.quats = nn.Parameter(quats)
        
        # 4. Opacities (Przezroczystość): Startujemy z prawie nieprzezroczystymi (logit(0.5) ~ 0.0).
        # Sigmoid(0.0) = 0.5. Optymalizator zdecyduje, które znikną.
        self.opacities = nn.Parameter(torch.zeros((self.N, 1), device=self.device))
        
        # 5. Colors (Spherical Harmonics - SH):
        # Dla uproszczenia (stopień 0) to po prostu diffuse color.
        # Konwertujemy RGB -> SH (uproszczona konwersja: (RGB - 0.5) / 0.28209).
        # EDUCATIONAL: SH stopnia 0 reprezentuje kolor niezależny od kąta (jak oświetlenie ambient).
        SH_C0_SCALE = 0.28209479177387814
        base_color = (self.colors_initial - 0.5) / SH_C0_SCALE
        self.sh0 = nn.Parameter(base_color.unsqueeze(1)) # [N, 1, 3]
        
        # Lista wszystkich parametrów do optymalizacji
        self.params = [self.means, self.scales, self.quats, self.opacities, self.sh0]

    def _setup_optimizer(self, lr=0.001):
        """Ustawia optymalizator Adam."""
        # Różne learning rates dla różnych parametrów (standardowe praktyki w 3DGS)
        self.optimizer = torch.optim.Adam([
            {'params': [self.means], 'lr': 0.00016 * 10}, # Pozycja zmienia się szybciej
            {'params': [self.colors_initial], 'lr': 0.0025},
            {'params': [self.scales], 'lr': 0.005},
            {'params': [self.quats], 'lr': 0.001},
            {'params': [self.sh0], 'lr': 0.0025},
            {'params': [self.opacities], 'lr': 0.05},
        ], lr=lr)

    def render(self, view_matrix: torch.Tensor, K: torch.Tensor, width: int, height: int) -> torch.Tensor:
        """
        Wykonuje Rasteryzację (Forward Pass).
        
        Args:
            view_matrix: Świat -> Kamera (4x4).
            K: Macierz intrisics kamery (3x3).
            width, height: Rozdzielczość obrazka.
            
        Returns:
            image: [H, W, 3] Wyrenderowany obraz RGB.
        """
        # EDUCATIONAL: Przygotowanie parametrów do rasteryzera
        scales_final = torch.exp(self.scales)
        quats_final = self.quats / self.quats.norm(dim=-1, keepdim=True)
        opacities_final = torch.sigmoid(self.opacities)
        
        # Przygotowanie kolorów (upraszczamy do RGB/SH0)
        SH_C0_SCALE = 0.28209479177387814
        colors_precomp = self.sh0.squeeze(1) * SH_C0_SCALE + 0.5
        
        # Gsplat oczekuje batch dimension dla kamer.
        # view_matrix: [1, 4, 4]
        # K: [1, 3, 3]
        if view_matrix.ndim == 2:
            view_matrix = view_matrix.unsqueeze(0)
        if K.ndim == 2:
            K = K.unsqueeze(0)

        # Rasteryzacja
        render_colors, render_alphas, info = rasterization(
            means=self.means,
            quats=quats_final,
            scales=scales_final,
            opacities=opacities_final.squeeze(-1),
            colors=colors_precomp, 
            viewmats=view_matrix,
            Ks=K,
            width=width,
            height=height
        )
        
        # Zwracamy pierwszy (i jedyny) obraz w batchu
        return render_colors[0]

    def optimize_step(self, gt_image: torch.Tensor, view_mat: torch.Tensor, K: torch.Tensor):
        """
        Wykonuje jeden krok treningowy (Iterację).
        """
        H, W, _ = gt_image.shape
        self.optimizer.zero_grad()
        
        # 1. Forward
        rendered_image = self.render(view_mat, K, W, H)
        
        # 2. Loss (L1 - błąd absolutny)
        # EDUCATIONAL: L1 Loss wymusza zgodność kolorów piksel w piksel.
        loss = torch.abs(rendered_image - gt_image).mean()
        
        # 3. Backward (Propagacja wsteczna)
        loss.backward()
        
        # 4. Update (Krok optymalizatora)
        self.optimizer.step()
        
        return loss.item(), rendered_image

    def save_ply(self, path: str):
        """
        Zapisuje wynik do binarnego pliku PLY kompatybilnego z przeglądarkami 3DGS (np. Antimatter15).
        Eksportujemy surowe parametry (przed aktywacją), tak jak w oryginalnej implementacji Inria.
        """
        xyz = self.means.detach().cpu().numpy()
        f_dc = self.sh0.detach().squeeze(1).cpu().numpy() # [N, 3]
        opacities = self.opacities.detach().cpu().numpy() # [N, 1] raw logits
        scales = self.scales.detach().cpu().numpy()       # [N, 3] raw logs
        rots = self.quats.detach().cpu().numpy()          # [N, 4] raw quats
        
        N = xyz.shape[0]
        
        # 1. Normals (fill with 0)
        normals = np.zeros_like(xyz)
        
        # 2. Construct structured array
        dtype_full = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
            ('opacity', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
        ]
        
        elements = np.empty(N, dtype=dtype_full)
        
        elements['x'] = xyz[:, 0]
        elements['y'] = xyz[:, 1]
        elements['z'] = xyz[:, 2]
        elements['nx'] = normals[:, 0]
        elements['ny'] = normals[:, 1]
        elements['nz'] = normals[:, 2]
        
        elements['f_dc_0'] = f_dc[:, 0]
        elements['f_dc_1'] = f_dc[:, 1]
        elements['f_dc_2'] = f_dc[:, 2]
        
        elements['opacity'] = opacities[:, 0]
        
        elements['scale_0'] = scales[:, 0]
        elements['scale_1'] = scales[:, 1]
        elements['scale_2'] = scales[:, 2]
        
        elements['rot_0'] = rots[:, 0]
        elements['rot_1'] = rots[:, 1]
        elements['rot_2'] = rots[:, 2]
        elements['rot_3'] = rots[:, 3]
        
        # 3. Write via plyfile
        el = plyfile.PlyElement.describe(elements, 'vertex')
        plyfile.PlyData([el]).write(path)
        
        print(f"[GaussianOptimizer] Zapisano binarny PLY do {path}")
