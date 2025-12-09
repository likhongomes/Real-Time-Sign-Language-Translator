from Utils.config import get_config
from Utils.dataset import create_data_loaders

cfg = get_config()
data_cfg = cfg["data"]
train_cfg = cfg["training"]

train_loader, val_loader, _ = create_data_loaders(data_cfg, train_cfg)

print("Train samples:", len(train_loader.dataset))
print("Val samples:  ", len(val_loader.dataset))

# Inspect what the val set actually is
val_ds = val_loader.dataset
print("First few val samples:")
for i in range(min(10, len(val_ds))):
    s = val_ds.samples[i]
    print(i, s["label"], s["video_path"])
