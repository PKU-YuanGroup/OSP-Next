import torch
from ospnext.modules.osp_next import OSPNextModel

# path
orig_weights_path = '/path/to/orig_torch_model'
save_path = '/path/to/diffusers_model'

# config, copy from configs
config = {
  'dim': 5120,
  'ffn_dim': 13824,
  'freq_dim': 256,
  'in_dim': 16,
  'num_heads': 40,
  'num_layers': 40,
  'out_dim': 16,
  'text_len': 512,
  'skiparse_model_type': "dual_end",
  'sparse_ratio': 2,
  'num_full_blocks': 8,
  'num_register_tokens': 0,
  'skiparse_1d': False,
  'skiparse_2d': True,
}

state_dict = torch.load(orig_weights_path, map_location='cpu')
model = OSPNextModel(**config)
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
print(f"missing_keys: {missing_keys} \nunexpected_keys: {unexpected_keys}")
model.save_pretrained(save_path)