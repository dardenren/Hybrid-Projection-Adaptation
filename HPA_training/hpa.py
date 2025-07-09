from typing import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class HpaModule(nn.Module):
    """
    HPA methods for modular low-rank updates on a base weight matrix.
    Supports multiple projection directions, initialization modes, and rank adjustments.
    """

    def __init__(
        self,
        base: nn.Module,
        rank: int,
        hpa_alpha: Optional[int],
        hpa_dropout: float,
        proj_direction: str,
        mode: str,
        eps_scale: float,
        adapt_weight_transpose: bool = False,
        **kwargs
    ):

        if proj_direction not in ["left", "right", "thin_left", "thin_right", "fat_right", "fat_left", "both"]:
            raise ValueError(f"Invalid projection direction: {proj_direction}")

        if mode not in ["svd", "random", "random_sample", "lora"]:
            valid_prefixes = ["random_svd", "power_sample"]
            if not any(mode.startswith(p) and mode[len(p):].isdigit() for p in valid_prefixes):
                raise ValueError(f"Invalid mode: {mode}")

        if rank <= 0:
            raise ValueError("Rank must be greater than 0")

        self.training = base.training
        self.weight = base.weight
        self.adapt_weight_transpose = adapt_weight_transpose
        if adapt_weight_transpose:
            self.cols, self.rows = self.weight.shape[:2]
        else:
            self.rows, self.cols = self.weight.shape[:2]

        if rank > (eff_rank := min(self.rows, self.cols) // 2):
            logger.info(f"Warning: input rank {rank} exceeds effective rank {eff_rank}, reduced to {eff_rank}")
            rank = eff_rank

        self.rank = rank
        self.mode = mode
        if mode in ["lora"]:
            self.proj_direction = "left"
        elif proj_direction in ["left", "right", "both"]:
            self.proj_direction = proj_direction
        elif self.rows <= self.cols:
            self.proj_direction = "right" if proj_direction in ["thin_left", "fat_right"] else "left"
        else:
            self.proj_direction = "left" if proj_direction in ["thin_left", "fat_right"] else "right"

        self.hpa_dropout = lambda x: x if not hpa_dropout else nn.Dropout(p=hpa_dropout)
        self.scaling = 1.0 if hpa_alpha is None else hpa_alpha / rank
        self.epscale = eps_scale * max(self.rows, self.cols)

        self.NONE_TENSOR = nn.Parameter(torch.empty(0, device=self.device), requires_grad=False)
        self._hpa_reset_and_disable()
        self.accumulated_rank = 0

    @property
    def dtype(self):
        return self.weight.dtype

    @property
    def device(self):
        return self.weight.device

    def param_repr(self) -> str:
        repr = f"mode={self.mode}, rank={self.rank}"
        return repr

    def attr_repr(self) -> str:
        repr = f"\n  weight: "
        if self.weight.requires_grad:
            repr += "Trainable\n"
        else:
            repr += "Frozen\n"

        if hasattr(self, "L"):
            repr += f"  L: "
            if self.L is self.NONE_TENSOR:
                repr += "Inactive\n"
            elif self.L.requires_grad:
                repr += "Trainable\n"
            else:
                repr += "Frozen\n"

        repr += f"  A: "
        if self.A is self.NONE_TENSOR:
            repr += "Inactive\n"
        elif self.A.requires_grad:
            repr += "Trainable\n"
        else:
            repr += "Frozen\n"

        if hasattr(self, "R"):
            repr += f"  R: "
            if self.R is self.NONE_TENSOR:
                repr += "Inactive\n"
            elif self.R.requires_grad:
                repr += "Trainable\n"
            else:
                repr += "Frozen\n"
        return repr

    def _hpa_reset_and_disable(self):
        """Sets adapters to effectively None."""
        if self.proj_direction == "left":
            self.L = self.NONE_TENSOR
            self.A = self.NONE_TENSOR
        elif self.proj_direction == "right":
            self.A = self.NONE_TENSOR
            self.R = self.NONE_TENSOR
        else:
            self.L = self.NONE_TENSOR
            self.A = self.NONE_TENSOR
            self.R = self.NONE_TENSOR

        self.hpa_enabled = False

    def reset_parameters(self):
        """Resets adapter parameters to initial state."""
        self._hpa_reset_and_disable()
        nn.Module.reset_parameters(self)

    def _hpa_forward(self, x: torch.Tensor, module_specific_forward: Callable = lambda u, P: u @ P) -> torch.Tensor:
        """Computes adapter output given input x."""
        if self.adapt_weight_transpose:
            if not self.hpa_enabled:
                return torch.zeros((*x.shape[:-1], self.cols), device=x.device, dtype=x.dtype)
        
        else:
            if not self.hpa_enabled:
                return torch.zeros((*x.shape[:-1], self.rows), device=x.device, dtype=x.dtype)

        if self.proj_direction == "left":
            return (module_specific_forward(x, self.A.T) @ self.L.T) * self.scaling
        elif self.proj_direction == "right":
            return (module_specific_forward(x, self.R.T) @ self.A.T) * self.scaling
        else:
            return (module_specific_forward(x, self.R.T) @ self.A.T @ self.L.T) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("forward() must be overriden by subclass")

    def _hpa_consolidate_weights(self) -> torch.Tensor:
        """Returns effective low-rank weight contribution."""
        if not self.hpa_enabled:
            res = torch.zeros((self.rows, self.cols), device=self.device, dtype=self.dtype)

        elif self.proj_direction == "left":
            res = (self.L.data @ self.A.data) * self.scaling
        elif self.proj_direction == "right":
            res = (self.A.data @ self.R.data) * self.scaling
        else:
            res = (self.L.data @ self.A.data @ self.R.data) * self.scaling

        return res.T if self.adapt_weight_transpose else res

    def _randomized_svd(self, data: torch.Tensor, q: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs randomized SVD with q power iterations."""
        with torch.no_grad():
            M = data.T if self.rows <= self.cols else data
            Y = M @ torch.randn(M.shape[1], self.rank, device=self.device, dtype=self.dtype)

            for _ in range(q):
                Y = M.T @ Y
                Y = M @ Y

            Q = torch.linalg.qr(Y)[0]

            if self.rows <= self.cols:
                U, S, VtQt = torch.linalg.svd(M @ Q.T, full_matrices=False)
                Vt = VtQt @ Q
            else:
                QtU, S, Vt = torch.linalg.svd(Q.T @ M, full_matrices=False)
                U = Q @ QtU

            return U, S, Vt

    def _hpa_activate(self, data: torch.Tensor):  # recalculates adapters and set to train mode
        if data.shape != self.weight.shape:
            raise ValueError(f"Invalid input shape: {data.shape}, expected {self.weight.shape}")
        
        if self.adapt_weight_transpose:
            data = data.T

        with torch.no_grad():  # Prevents gradients during initialization
            data.to(self.device, self.dtype)  # Ensure data is on correct device and dtype

            if self.mode not in ["lora"]:  # QR/SVD-based initialization for most modes

                if self.mode == "svd":  # Full deterministic SVD
                    U, D, Vt = torch.linalg.svd(data, full_matrices=False)
                    L = U[:, :self.rank]  # Left singular vectors
                    R = Vt[:self.rank, :]  # Right singular vectors

                elif "random_svd" in self.mode:  # Approximate randomized SVD with power iterations
                    L, D, R = self._randomized_svd(data, int(self.mode[10:]))

                elif self.mode == "random":
                    # Fully random orthogonal initialization
                    if self.proj_direction != "right":
                        L = torch.linalg.qr(torch.randn((self.rows, self.rank), device=self.device, dtype=self.dtype))[0]
                    if self.proj_direction != "left":
                        R = torch.linalg.qr(torch.randn((self.cols, self.rank), device=self.device, dtype=self.dtype))[0].T

                elif self.mode == "random_sample":
                    # Randomized initialization based on rows/columns
                    if self.proj_direction != "right":
                        L = torch.linalg.qr(data @ torch.randn((self.cols, self.rank), device=self.device, dtype=self.dtype))[0]
                    if self.proj_direction != "left":
                        R = torch.linalg.qr(data.T @ torch.randn((self.rows, self.rank), device=self.device, dtype=self.dtype))[0].T

                elif "power_sample" in self.mode:
                    # Similar to randomized SVD, but avoids final SVD step
                    if self.proj_direction != "right":
                        Y = torch.randn((self.rows, self.rank), device=self.device, dtype=self.dtype)
                        for i in range(int(self.mode[12:])):  # repeated power iterations
                            Y = data.T @ Y
                            Y = data @ Y
                        L = torch.linalg.qr(Y)[0]
                    if self.proj_direction != "left":
                        Y = torch.randn((self.cols, self.rank), device=self.device, dtype=self.dtype)
                        for i in range(int(self.mode[12:])):
                            Y = data @ Y
                            Y = data.T @ Y
                        R = torch.linalg.qr(Y)[0].T

                # Final parameter assignment based on projection direction
                if self.proj_direction == "left":
                    self.L = nn.Parameter(L, requires_grad=False)
                    self.A = nn.Parameter(L.T @ data, requires_grad=True)  # Such that L A = L L.T data (projection of data onto left subspace)

                elif self.proj_direction == "right":
                    self.R = nn.Parameter(R, requires_grad=False)
                    self.A = nn.Parameter(data @ R.T, requires_grad=True)  # Right subspace

                else:
                    self.L = nn.Parameter(L, requires_grad=False)
                    self.R = nn.Parameter(R, requires_grad=False)
                    self.A = nn.Parameter(L.T @ data @ R.T, requires_grad=True)  # Double projection

            else:  # Non-QR based initialization, e.g., LoRA
                if self.mode == "lora":
                    self.L = nn.Parameter(torch.zeros((self.rows, self.rank), device=self.device, dtype=self.dtype), requires_grad=True)
                    self.A = nn.Parameter(torch.randn((self.rank, self.cols), device=self.device, dtype=self.dtype), requires_grad=True)

        self.hpa_enabled = True

    def _hpa_checkrank_activate(self, data: torch.Tensor):
        """
        Recalculates adapters while calculating effective rank via SVD.
        Reduces self.rank if applicable.
        """
        if data.shape != self.weight.shape:
            raise ValueError(f"Invalid input shape: {data.shape}, expected {self.weight.shape}")

        with torch.no_grad():
            if self.mode == "svd":
                S = torch.linalg.svd(self.weight.grad, full_matrices=False)[1]
            else:  # Default to randomized SVD with 2 power iterations for efficiency
                S = self._randomized_svd(self.weight.grad, 2)[1]

            # Threshold based on machine epsilon and scale to detect near-zero singular values
            cutoff = self.epscale * torch.finfo(self.dtype).eps * torch.max(data)
            effective_rank = (S > cutoff).sum().item()
            self.rank = min(effective_rank, self.rank)  # Adjust rank
            self._hpa_activate(data)

        self.hpa_enabled = True

    def train(self, mode: bool = True):
        if not mode:
            self.merge_weights()

        nn.Module.train(self, mode)

    def merge_weights(self):
        """Merges adapters to weights and disables adapters."""
        if self.training and self.hpa_enabled:
            with torch.no_grad():
                self.weight += self._hpa_consolidate_weights()
                self.accumulated_rank += self.rank
                self._hpa_reset_and_disable()
                self.weight.requires_grad = True

    def reactivate_on_input(self, data: torch.Tensor, checkrank: bool = False):
        """Reinitializes adapter based on new input, with optional rank check."""
        if self.training and not self.hpa_enabled:
            with torch.no_grad():
                self.weight.requires_grad = False
                if checkrank:
                    self._hpa_checkrank_activate(data)
                else:
                    self._hpa_activate(data)
                self.weight -= self._hpa_consolidate_weights()

    def reactivate_on_weight(self, checkrank: bool = False):
        """Reinitializes adapter based on stored weight matrix."""
        self.reactivate_on_input(self.weight.data, checkrank)

    def reactivate_on_gradient(self, checkrank: bool = False):
        """Reinitializes adapter based on gradient matrix."""
        self.reactivate_on_input(self.weight.grad, checkrank)

    def accumulated_size(self, fn: Callable[[int, int, int], int]) -> int:
        """Computes storage cost given size function fn(m, n, rank)."""
        return fn(self.cols, self.rows, self.accumulated_rank)

    def get_side_adapter(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        L = self.L.data if (self.hpa_enabled and hasattr(self, "L")) else None
        R = self.R.data if (self.hpa_enabled and hasattr(self, "R")) else None
        return L, R

    def get_side_transpose(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        Lt = self.L.data.T if (self.hpa_enabled and hasattr(self, "L")) else None
        Rt = self.R.data.T if (self.hpa_enabled and hasattr(self, "R")) else None
        return Lt, Rt


class HpaLinear(HpaModule, nn.Linear):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        hpa_alpha: Optional[int] = 16,
        hpa_dropout: float = 0.,
        proj_direction: str = "thin_left",
        mode: str = "svd",
        eps_scale: float = 1.,
        **kwargs
    ):
        if not isinstance(base, nn.Linear):
            raise ValueError("Base module must be nn.Linear")

        # Initialize nn.Linear using base's shape. Goes first as all existing params get cleared upon nn.Module.__init__()
        nn.Linear.__init__(self, in_features=base.in_features, out_features=base.out_features, bias=(base.bias is not None))

        # Initialize HpaModule
        HpaModule.__init__(self, base, rank, hpa_alpha, hpa_dropout, proj_direction, mode, eps_scale)
        self.bias = base.bias

    def __repr__(self):
        repr = super(nn.Linear, self).__repr__()
        if "\n" in repr:
            split = repr.split("\n", 2)
            return split[0] + "\n" + split[1] + ", " + self.param_repr() + split[2].rsplit("\n", 1)[0] + "\n)"
        else:
            split = repr.rsplit(")", 1)
            return split[0] + ", " + self.param_repr() + ")"

    def reset_parameters(self) -> None:
        nn.Linear.reset_parameters(self)

        if hasattr(self, "A"): # Other class may call self.reset_parameters() before HpaModule.__init__()
            HpaModule.reset_parameters(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass combining base weight and adapter output."""
        if self.hpa_enabled: # F.linear transposes weights internally
            return nn.Linear.forward(self, x) + self._hpa_forward(self.hpa_dropout(x))

        return nn.Linear.forward(self, x)

    def train(self, mode: bool = True) -> None:
        HpaModule.train(self, mode)
        nn.Linear.train(self, mode)


class HpaEmbedding(HpaModule, nn.Embedding):
    def __init__(
        self,
        base: nn.Embedding,
        rank: int,
        hpa_alpha: Optional[int] = 16,
        proj_direction: str = "thin_left",
        mode: str = "svd",
        eps_scale: float = 1.,
        **kwargs
    ):
        if not isinstance(base, nn.Embedding):
            raise ValueError("Base module must be nn.Embedding")

        # Initialize nn.Embedding()
        nn.Embedding.__init__(self, num_embeddings=base.num_embeddings, embedding_dim=base.embedding_dim, \
                              padding_idx=base.padding_idx, max_norm=base.max_norm, norm_type=base.norm_type, \
                              scale_grad_by_freq=base.scale_grad_by_freq, sparse=base.sparse)

        # Initialize HpaModule
        HpaModule.__init__(self, base, rank, hpa_alpha, None, proj_direction, mode, eps_scale, adapt_weight_transpose=True)

    def __repr__(self):
        repr = super(nn.Embedding, self).__repr__()
        if "\n" in repr:
            split = repr.split("\n", 2)
            return split[0] + "\n" + split[1] + ", " + self.param_repr() + split[2].rsplit("\n", 1)[0] + "\n)"
        else:
            split = repr.rsplit(")", 1)
            return split[0] + ", " + self.param_repr() + ")"

    def reset_parameters(self) -> None:
        nn.Embedding.reset_parameters(self)

        if hasattr(self, "A"): # Other class may call self.reset_parameters() before HpaModule.__init__()
            HpaModule.reset_parameters(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass combining base weight and adapter output."""
        def apply_fn(u, P):
            return F.embedding(u, P, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

        if self.hpa_enabled:
            return nn.Embedding.forward(self, x) + self._hpa_forward(x, apply_fn)

        return nn.Embedding.forward(self, x)

    def train(self, mode: bool = True) -> None:
        HpaModule.train(self, mode)
        nn.Embedding.train(self, mode)


def make_flexi_optimizer(optimizer: Type[optim.Optimizer], target_states: List[str]) -> optim.Optimizer:
    """
    Wraps an optimizer to support state projection for HPA layers.
    Allows transitioning optimizer state between full weights (weight) and adapters (A),
    while managing internal parameter lists and placeholder tensors.
    """

    class FlexiStateOptimizer(optimizer):
        def __init__(self, model: nn.Module, params: Iterable[Union[Tuple[str, torch.Tensor], Dict]], **kwargs):
            """
            Initializes optimizer with special handling for HPA modules.

            - Expects params to be an iterable of (name, param) tuples, or dicts with the key 'params' containing (name, param) tuples.
            - Detects HPA modules and duplicates their 'weight' param slots with corresponding 'A' placeholders.
            - Stores placeholders as empty tensors to satisfy optimizer param constraints.
            """
            params = [*params]
            if not isinstance(params[0], dict):
                params = [{'params': params}]

            self.hpa_modules = {}

            for pg_no, pg in enumerate(params):
                params_list = []      # Holds standard parameters
                hpa_params_list = []  # Holds the current trainable parameter for HPA, alternating between weight and A
                hpa_index = 0

                for name, param in pg['params']:
                    name = name.rsplit('.', 1)
                    module = model.get_submodule(name[0])

                    if isinstance(module, HpaModule) and name[1] == 'weight':
                        self.hpa_modules[module] = (pg_no, hpa_index)
                        if module.mode not in ["lora"]:
                            hpa_params_list.append(param)
                            hpa_index += 1

                        else:
                            hpa_params_list.extend([param, torch.empty(0, device=module.device)]) # Extra slot for unfrozen L
                            hpa_index += 2

                    elif param.requires_grad:
                        params_list.append(param)

                pg['params'] = hpa_params_list + params_list

            self.hpa_enabled = False  # Tracks whether optimizer is in adapter mode (True) or full weights mode (False)
            self.target_states = target_states[:]
            optimizer.__init__(self, params, **kwargs)

        def _set_hpa_param(self, module: HpaModule, value: nn.Parameter, offset: int = 0) -> None:
            """Replaces the weight placeholder for the HPA module at given index."""
            pg_no, hpa_index = self.hpa_modules[module]
            self.param_groups[pg_no]['params'][hpa_index + offset] = value

        def project_states(self):
            """
            Projects optimizer state values from full weights to corresponding adapters (A).

            - Moves target states (e.g., momentum, variance) from weight to A.
            - Applies the relevant transformations to each state during transfer.
            - Clears weight states and swaps optimizer param to A.
            """
            if self.hpa_enabled:
                return

            for module in self.hpa_modules:
                if module.mode not in ["lora"]: # Modes that require projection
                    state_dict = self.state.pop(module.weight, {})

                    for target_state in self.target_states:
                        if target_state in state_dict:
                            temp = state_dict[target_state]
                            side_transpose = module.get_side_transpose()

                            with torch.no_grad():
                                if side_transpose[0] is not None:
                                    temp = side_transpose[0] @ temp
                                if side_transpose[1] is not None:
                                    temp = temp @ side_transpose[1]

                            state_dict[target_state] = temp

                else: # Modes that do not require projection. Currently only LoRA is supported
                    state_dict = {}
                    self._set_hpa_param(module, module.L, 1)

                self._set_hpa_param(module, module.A)
                self.state[module.A] = state_dict

            self.zero_grad()
            self.hpa_enabled = True

        def merge_states(self):
            """
            Merges optimizer state values from adapters (A) back to full weights.

            - Moves target states from A to weight.
            - Applies the relevant transformations to each state during transfer.
            - Clears A states and swaps optimizer param to weight.
            """
            if not self.hpa_enabled:
                return

            for module in self.hpa_modules:
                if module.mode not in ["lora"]: # Modes that require projection
                    state_dict = self.state.pop(module.A, {})

                    for target_state in self.target_states:
                        if target_state in state_dict:
                            temp = state_dict[target_state]
                            side_adapter = module.get_side_adapter()

                            with torch.no_grad():
                                if side_adapter[0] is not None:
                                    temp = side_adapter[0] @ temp
                                if side_adapter[1] is not None:
                                    temp = temp @ side_adapter[1]

                            state_dict[target_state] = temp

                else: # Modes that do not require projection
                    state_dict = {}
                    self.state.pop(module.L, None) # Important: free memory

                self._set_hpa_param(module, module.weight)
                self.state[module.weight] = state_dict

            self.zero_grad()
            self.hpa_enabled = False

    return FlexiStateOptimizer

def replace_with_hpa(model: nn.Module, target_modules: List[str] = [""], set_rank_fn: Callable = lambda m, n: max(16, min(m, n)//32), **kwargs):
    '''
    Targets nn.Linear and nn.Embedding modules in the model and replaces them with HpaLinear and HpaEmbedding.

    set_rank_fn: function that maps (rows, columns) to target rank
    '''
    modules_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if any(target_key in name for target_key in target_modules):
                if '.' not in name:
                    parent_module = model
                    child_name = name
                else:
                    parent_path, child_name = name.rsplit('.', 1)
                    parent_module = model.get_submodule(parent_path)

                if isinstance(module, (HpaLinear, HpaEmbedding)):
                    module.merge_weights()
                    hpa_type = HpaLinear if isinstance(module, HpaLinear) else HpaEmbedding
                    rank = set_rank_fn(module.rows, module.cols)
                    eff_r = min(module.rows, module.cols) // 2 # Only apply HPA if memory cost with given rank is strictly less than full FT

                else:
                    if isinstance(module, nn.Linear):
                        hpa_type = HpaLinear
                        rank = set_rank_fn(module.out_features, module.in_features)
                        eff_r = min(module.out_features, module.in_features) // 2

                    elif isinstance(module, nn.Embedding):
                        hpa_type = HpaEmbedding
                        rank = set_rank_fn(module.embedding_dim, module.num_embeddings)
                        eff_r = min(module.embedding_dim, module.num_embeddings) // 2

                if rank < eff_r:
                    modules_to_replace.append((parent_module, child_name, module, hpa_type, rank))

    for parent_module, child_name, original_linear_module, hpa_type, rank in modules_to_replace:
        setattr(parent_module, child_name, hpa_type(original_linear_module, rank, **kwargs))