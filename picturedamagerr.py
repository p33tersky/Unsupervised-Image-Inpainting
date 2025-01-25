import torch
import torch.nn.functional as F
import numpy as np

class PictureDamager:
    def __init__(self, masks: list, stain_percentage: float):
        """
        masks: lista 2D-tensorów (numpy.ndarray lub torch.Tensor)
        stain_percentage: ułamek określający, jaki procent pikseli ma być "pobrudzony"
        """
        self.stain_percentage = stain_percentage

        # Konwertuj wszystkie maski na tensory PyTorch
        self.masks = [
            torch.from_numpy(mask).to(dtype=torch.uint8, device='cuda')
            if isinstance(mask, np.ndarray) else mask.to(dtype=torch.uint8, device='cuda')
            for mask in masks
        ]

    def expected_num_of_stains(self, mask_size):
        # mask_size to krotka (wysokość, szerokość)
        return mask_size[0] * mask_size[1] * self.stain_percentage

    def _morphological_dilation(self, mask: torch.Tensor) -> torch.Tensor:
        kernel = torch.ones((1, 1, 3, 3), device='cuda', dtype=torch.uint8)  # Poprawione dtype
        mask_f = mask.unsqueeze(0).unsqueeze(0).float()   
        out = F.conv2d(mask_f, kernel.float(), padding=1)
        out = (out > 0).float()
        return out[0, 0]

    def _morphological_erosion(self, mask: torch.Tensor) -> torch.Tensor:
        kernel = torch.ones((1, 1, 3, 3), device='cuda', dtype=torch.uint8)  # Poprawione dtype
        mask_f = mask.unsqueeze(0).unsqueeze(0).float()
        out = F.conv2d(mask_f, kernel.float(), padding=1)
        out = (out == kernel.numel()).float()
        return out[0, 0]
    
    def mask_randomizer(self, mask: torch.Tensor, resize_prob=0.5, morph_transform_prob=0.5):
        """
        Odpowiednik wersji numpy/scipy, ale w PyTorchu.
        """
        # Losowe powiększenie/pomniejszenie
        if torch.rand(()).item() < resize_prob:
            enlarge = (torch.rand(()).item() < 0.5)
            if enlarge:
                if torch.rand(()).item() < 0.5:
                    # powiększ x2
                    mask = F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0).float(),
                        scale_factor=(2, 2),
                        mode='nearest'
                    ).squeeze(0).squeeze(0)
                else:
                    # powiększ x3
                    mask = F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0).float(),
                        scale_factor=(3, 3),
                        mode='nearest'
                    ).squeeze(0).squeeze(0)
            else:
                # pomniejsz x0.5
                mask = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),
                    scale_factor=(0.5, 0.5),
                    mode='nearest'
                ).squeeze(0).squeeze(0)

        # Losowa operacja morfologiczna
        if torch.rand(()).item() < morph_transform_prob:
            dilate = (torch.rand(()).item() < 0.5)
            if dilate:
                mask = self._morphological_dilation(mask)
            else:
                mask = self._morphological_erosion(mask)

        # mask ma w tym momencie typ float (0.0/1.0).  
        # Możemy wymusić 0/1 (zaokrąglić)
        mask = (mask >= 0.5).to(torch.uint8)

        # Transpozycja
        if torch.rand(()).item() < 0.5:
            # ekwiwalent mask = mask.T
            mask = mask.transpose(0, 1)

        # Flip up-down
        if torch.rand(()).item() < 0.5:
            mask = torch.flip(mask, dims=[0])

        # Flip left-right
        if torch.rand(()).item() < 0.5:
            mask = torch.flip(mask, dims=[1])

        return mask

    def generate_random_mask(self, mask_size=(256, 256)):
        """
        Zwraca 2D-tensor o wymiarach mask_size (H,W) z wartościami 0/1,
        w którym „naniesione” zostają plamki (stains) wybrane z puli self.masks.
        """
        device = 'cuda'  # zakładamy, że wszystkie maski są na tym samym device
        damaged_mask = torch.zeros(mask_size, dtype=torch.uint8, device=device)
        stains_counter = 0

        target_stains = self.expected_num_of_stains(mask_size)

        while stains_counter < target_stains:
            # Losowa maska z puli
            idx = torch.randint(low=0, high=len(self.masks), size=()).item()
            mask = self.masks[idx]

            # Przetwarzamy maskę (powiększenie/pomniejszenie, flippy, itd.)
            mask = self.mask_randomizer(mask, 0.5, 0.5)

            mask_height, mask_width = mask.shape
            # Jeśli maska jest większa od obszaru docelowego, pomiń
            if mask_height > mask_size[0] or mask_width > mask_size[1]:
                continue

            # Losowe położenie w docelowej masce
            x = torch.randint(low=0, high=(mask_size[0] - mask_height + 1), size=()).item()
            y = torch.randint(low=0, high=(mask_size[1] - mask_width + 1), size=()).item()

            # Wklejenie maski binarnej (tam gdzie mask==1, nadpisujemy damaged_mask)
            region = damaged_mask[x : x + mask_height, y : y + mask_width]
            damaged_mask[x : x + mask_height, y : y + mask_width] = torch.where(
                mask == 1, 
                torch.tensor(1, device=device, dtype=torch.uint8), 
                region
            )

            stains_counter += mask.sum().item()

        return damaged_mask