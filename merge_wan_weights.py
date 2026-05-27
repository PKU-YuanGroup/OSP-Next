import torch
from ospnext.modules.want2v import WanModel

orig_path = "Wan2.1-T2V-14B"
save_path = "Wan2.1-T2V-14B/want2v_14b.pt"
model = WanModel.from_pretrained(orig_path)
state_dict = model.state_dict()
torch.save(state_dict, save_path)