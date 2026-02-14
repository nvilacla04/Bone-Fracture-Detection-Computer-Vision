#based on findings from prevous files 

from train import train
train(
    batch_size=16,
    head_epochs=4,
    finetune_epochs=9,
    lr_head=1e-4,
    lr_backbone=9.5e-5,
)
