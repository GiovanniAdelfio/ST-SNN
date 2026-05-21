import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
from .laplacian_builder import GeneralLaplacianBuilder


class SheafNN(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 edge_index: torch.Tensor,
                 num_nodes: int = 25,
                 n_layers: int = 2,
                 ort: bool = True,
                 stalk: int = 2,
                 act: str = 'F.elu',
                 device="cuda",
                 res = False):
        super().__init__()
        assert out_channels % stalk == 0, "out_channels must be divisible by the stalk dimension"

        self.device = device
        self.out_channels = out_channels
        self.stalk = stalk
        self.n_layers = n_layers
        self.num_nodes = num_nodes
        self.act = eval(act)
        self.orthogonal = ort
        self.with_res = res

        # Conv2d(kernel=1) è equivalente a Linear ma opera nativamente su (B, C, T, V)
        # senza nessun permute o reshape
        self.lin_in = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

        # Rimuove i self-loops
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]

        # Topologia statica
        self.laplacian_builder = GeneralLaplacianBuilder(
            size=num_nodes, edge_index=edge_index, d=stalk, normalised=False, deg_normalised=True
        )
        self._init_maps(edge_index)

        # Pesi per la diffusione: W1 su stalk, W2 su f
        # Anche questi diventano Conv per coerenza con il layout (B, C, T, V)
        f = out_channels // stalk
        self.W1_layers = nn.ModuleList()
        self.W2_layers = nn.ModuleList()
        for _ in range(n_layers):
            # W1 opera su stalk: Conv1d su dim stalk, condivisa su (B, f, T, V)
            self.W1_layers.append(nn.Linear(stalk, stalk, bias=False))
            # W2 opera su f: Conv2d(f, f, 1) opera su (B*stalk, f, T, V) 
            self.W2_layers.append(nn.Conv2d(f, f, kernel_size=1, bias=False))

        if self.with_res:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels))
            else:
                self.down = lambda x: x

    def _init_maps(self, edge_index):
        if self.stalk != 2 or not self.orthogonal:
            num_edges = edge_index.size(1)
            base_eye = torch.eye(self.stalk)
            stacked_eyes = base_eye.unsqueeze(0).expand(num_edges, self.stalk, self.stalk).clone()
            self.maps = nn.Parameter(stacked_eyes)
        else:
            # Un angolo per arco — parametrizzazione minimale per stalk=2
            self.maps = nn.Parameter(torch.zeros(edge_index.size(1)))

    def _build_laplacian(self):
        if self.orthogonal and self.stalk != 2:
            ort_maps = torch.matrix_exp(self.maps - self.maps.transpose(1, 2))
            laplacian, _ = self.laplacian_builder(ort_maps)
        elif self.orthogonal:
            c = torch.cos(self.maps)
            s = torch.sin(self.maps)
            ort_maps = torch.zeros(self.maps.size(0), 2, 2, device=self.maps.device)
            ort_maps[:, 0, 0] =  c
            ort_maps[:, 0, 1] = -s
            ort_maps[:, 1, 0] =  s
            ort_maps[:, 1, 1] =  c
            laplacian, _ = self.laplacian_builder(ort_maps)
        else:
            laplacian, _ = self.laplacian_builder(torch.tanh(self.maps))

        index, value = laplacian
        V = self.num_nodes
        L_dense = torch.zeros(V * self.stalk, V * self.stalk, device=value.device)
        L_dense[index[0], index[1]] = value
        return L_dense

    def forward(self, x):
        # x shape: (B, C_in, T, V) — nativo STGCN, nessun permute in ingresso
        B, C, T, V = x.size()
        f = self.out_channels // self.stalk
        
        res = self.down(x) if self.with_res else 0
        # Proiezione canali: Conv2d(kernel=1) su (B, C_in, T, V) → (B, C_out, T, V)
        x = self.lin_in(x)

        # Costruzione Laplaciano: (V*stalk, V*stalk)
        L_dense = self._build_laplacian()

        # Reshape per esporre stalk: (B, C_out, T, V) → (B, stalk, f, T, V)
        x_reshaped = x.view(B, self.stalk, f, T, V)

        for layer in range(self.n_layers):
            W1 = self.W1_layers[layer]
            W2 = self.W2_layers[layer]

            # W2 su f: lavora su (B*stalk, f, T, V) — Conv2d nativo, zero permute
            x_for_W2 = x_reshaped.view(B * self.stalk, f, T, V)
            x_W2 = W2(x_for_W2).view(B, self.stalk, f, T, V)   # → (B, stalk, f, T, V)

            # W1 su stalk: einsum su dim 1
            # (stalk_out, stalk_in) x (B, stalk_in, f, T, V) → (B, stalk_out, f, T, V)
            H_tensor = torch.einsum('ji, biktv -> bjktv', W1.weight, x_W2)

            # Diffusione Laplaciana:
            # (B, stalk, f, T, V) → (B, T, V*stalk, f)  [un solo permute, necessario per L]
            H_flat = H_tensor.permute(0, 3, 4, 1, 2).reshape(B, T, V * self.stalk, f)

            # L_dense: (V*stalk, V*stalk) x (B, T, V*stalk, f) → (B, T, V*stalk, f)
            H_out_flat = torch.einsum('ij, btjf -> btif', L_dense, H_flat)

            # Ritorno a (B, stalk, f, T, V)
            H_out = H_out_flat.view(B, T, V, self.stalk, f).permute(0, 3, 4, 1, 2)

            x_reshaped = x_reshaped - self.act(H_out)

        # Ricompatta: (B, stalk, f, T, V) → (B, C_out, T, V)
        out = x_reshaped.reshape(B, self.out_channels, T, V)

        # BN2d su (B, C, T, V) — nativo, nessun permute
        return self.bn(out) + res 
