import torch.nn as nn
import torch


class FeedForwardBlock(nn.Module):

    def __init__(self, n_dim) -> None:
        super().__init__()
        self.linear1 = nn.Linear(n_dim, n_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(n_dim, n_dim)

    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class AttentionBlock(nn.Module):

    def __init__(self, n_dim: int, n_head=8) -> None:
        super().__init__()
        self.n_head = n_head
        self.n_dim = n_dim
        self.attention = nn.MultiheadAttention(self.n_dim, self.n_head)
        self.ffn = FeedForwardBlock(self.n_dim)
        self.layer_norm1 = nn.LayerNorm(self.n_dim)
        self.layer_norm2 = nn.LayerNorm(self.n_dim)

    def forward(self, x: torch.Tensor):
        pass


class PetNet(nn.Module):

    def __init__(self, n_att=6, d_model=512, n_head=8, n_class=2) -> None:
        self.n_att = n_att
        self.d_model = d_model
        self.n_head = n_head
        self.n_class = n_class
        self.attention_blocks = nn.Sequential(
            *[AttentionBlock(self.d_model, self.n_head) for _ in range(self.n_att)]
        )
        self.cls_mlp = nn.Linear(self.d_model, n_class)
        # set div_item, no grad, but Parameter, for using device
        self.positional_encoding_div_item = nn.Parameter(...)
        self.image_embedding = nn.Sequential(...)  # Using a Linear mlp

    def image_embedding(self, x: torch.Tensor):
        pass

    def forward(self, x: torch.Tensor):
        """
        Assume x is a tensor of shape [batch_size, channel, height, width] or [channel, height, width].
            * If the shape is [channel, height, width], x = x.unsqueeze(0), add a batch dimension
            * x **MUST** be 3 * 224 * 224 image, this will be ensured by the dataset class. 
        Process:

        1. x = self.slice_image(x):
            [batch_size, 3, 224, 224] -> [batch_size, 3, (14 * 14) patch, where each patch is 16 * 16]
        2. x = self.image_embedding(x):
            [batch_size, 3, (14 * 14), 16 * 16] -> [batch_size, (14 * 14), self.d_model] convert each patch to a 512 dim vector
        3. x = self.concat_cls_token(x):
            [batch_size, (14 * 14), self.d_model] -> [batch_size, (14 * 14 + 1), self.d_model] add a cls token at the beginning
        4. x = self.positional_encoding(x):
            [batch_size, (14 * 14 + 1), self.d_model] -> [batch_size, (14 * 14 + 1), self.d_model] add positional encoding
        5. x = self.attention_blocks(x):
            [batch_size, (14 * 14 + 1), self.d_model] -> [batch_size, (14 * 14 + 1), self.d_model] apply attention blocks
        6. x = self.get_cls_token(x):
            [batch_size, (14 * 14 + 1), self.d_model] -> [batch_size, self.d_model] get the cls token
        7. x = self.cls_mlp(x):
            [batch_size, self.d_model] -> [batch_size, self.n_class] get the class logits
        self.d_model always equals 512, self.n_class always equals 2
        Args:
            x (torch.Tensor): _description_
        """
        pass

    def concat_cls_token(self, x: torch.Tensor):
        """
        Add a cls token at the beginning of the tensor. x is a tensor of shape [batch_size, n_patch, d_model]
        1. create a tensor of shape [batch_size, 1, d_model] filled with 1.

        [batch_size, n_patch, d_model] -> [batch_size, n_patch + 1, d_model]
        Args:
            x (torch.Tensor): _description_
        """
        pass

    def get_cls_token(self, x: torch.Tensor):
        """
        Get the cls token from the tensor. x is a tensor of shape [batch_size, n_patch + 1, d_model]
        [batch_size, n_patch + 1, d_model] -> [batch_size, d_model]
        Args:
            x (torch.Tensor): _description_
        """
        pass

    def slice_image(self, x: torch.Tensor):
        """
        Slice the image into patches. x is a tensor of shape [batch_size, 3, 224, 224]
        [batch_size, 3, 224, 224] -> [batch_size, 3, 14 * 14, 16 * 16]
        Args:
            x (torch.Tensor): _description_
        """
        pass

    def positional_encoding(self, x: torch.Tensor):
        """
        Apply positional encoding to the tensor. x is a tensor of shape [batch_size, n_patch + 1, d_model]
        [batch_size, n_patch + 1, d_model] -> [batch_size, n_patch + 1, d_model]
        Args:
            x (torch.Tensor): _description_
        """
        pass
