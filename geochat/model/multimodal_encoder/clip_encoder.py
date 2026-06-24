import torch
import torch.nn as nn
import math
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def _get_clip_embeddings(self):
        """Return the embeddings module, handling both transformers 4.x and 5.x structures."""
        vt = self.vision_tower
        if hasattr(vt, 'vision_model'):
            return vt.vision_model.embeddings
        return vt.embeddings

    def clip_interpolate_embeddings(self, image_size=600, patch_size=14):
        """Interpolate positional embeddings to support a different input resolution."""
        embeddings = self._get_clip_embeddings()
        state_dict = embeddings.position_embedding.state_dict()
        pos_embedding = state_dict['weight']
        pos_embedding = pos_embedding.unsqueeze(0)
        n, seq_length, hidden_dim = pos_embedding.shape
        if n != 1:
            raise ValueError(f"Unexpected position embedding shape: {pos_embedding.shape}")

        new_seq_length = (image_size // patch_size) ** 2 + 1

        if new_seq_length != seq_length:
            seq_length -= 1
            new_seq_length -= 1
            pos_embedding_token = pos_embedding[:, :1, :]
            pos_embedding_img = pos_embedding[:, 1:, :]

            pos_embedding_img = pos_embedding_img.permute(0, 2, 1)
            seq_length_1d = int(math.sqrt(seq_length))
            torch._assert(seq_length_1d * seq_length_1d == seq_length, "seq_length is not a perfect square!")

            pos_embedding_img = pos_embedding_img.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)
            new_seq_length_1d = image_size // patch_size

            new_pos_embedding_img = nn.functional.interpolate(
                pos_embedding_img,
                size=new_seq_length_1d,
                mode='bicubic',
                align_corners=True,
            )

            new_pos_embedding_img = new_pos_embedding_img.reshape(1, hidden_dim, new_seq_length)
            new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)
            new_pos_embedding = torch.cat([pos_embedding_token, new_pos_embedding_img], dim=1)[0]
            state_dict['weight'] = new_pos_embedding
            embeddings.position_embedding = nn.Embedding(new_seq_length + 1, hidden_dim)
            embeddings.position_embedding.load_state_dict(state_dict)
            embeddings.position_embedding.requires_grad_(False)
            embeddings.image_size = image_size
            embeddings.patch_size = patch_size
            embeddings.position_ids = torch.arange(new_seq_length + 1).expand((1, -1))
            
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
            # Load CLIP on CPU first to avoid conflicts with accelerate meta-device context
            import torch, contextlib
            ctx = torch.device("cpu") if hasattr(torch, "device") else contextlib.nullcontext()
            with contextlib.ExitStack() as stack:
                try:
                    stack.enter_context(torch.device("cpu"))
                except Exception:
                    pass
                self.vision_tower = CLIPVisionModel.from_pretrained(
                    self.vision_tower_name, device_map=None
                )
            self.vision_tower.requires_grad_(False)
            self.clip_interpolate_embeddings(image_size=504, patch_size=14)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)
        self.clip_interpolate_embeddings(image_size=504, patch_size=14)

        self.is_loaded = True
        # print(self.is_loaded)

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                # print(image.shape)
                # import pdb; pdb.set_trace()
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                # print(image_features.shape)
                
                image_features.append(image_feature)
        else:
            # print(images.shape)
            # import pdb; pdb.set_trace()
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            # print(image_features.shape)
            

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
