#Import packages
import math
from pytorch_lightning import LightningModule
import re
from torch import Tensor
import torch
import copy
from typing import List, Optional, Tuple
from timm.models.vision_transformer import vit_small_patch16_224
from torch import Tensor
from torch.optim import AdamW

from lightly.loss import DINOLoss, IBOTPatchLoss, KoLeoLoss
from lightly.models.modules import DINOProjectionHead, MaskedVisionTransformerTIMM
from lightly.models.utils import (
    random_block_mask,
    update_drop_path_rate,
    update_momentum,
)
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import datasets
from lightly.transforms import DINOTransform
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule
from timm.models.vision_transformer import vit_small_patch16_224
#Define DINO models, forward and losses
class DINOv2(LightningModule):
    def __init__(self, batch_size_per_device: int, num_classes: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size_per_device = batch_size_per_device
        ibot_separate_head = False

        # Teacher
        vit = vit_small_patch16_224(
            pos_embed="learn", dynamic_img_size=True, init_values=1e-5
        )
        self.backbone = MaskedVisionTransformerTIMM(
            vit=vit, antialias=False, pos_embed_initialization="skip"
        )
        self.dino_head = DINOProjectionHead(input_dim=384, norm_last_layer=False)
        self.ibot_head = self.dino_head
        if ibot_separate_head:
            self.ibot_head = DINOProjectionHead(input_dim=384, norm_last_layer=False)

        # Student
        self.student_backbone = copy.deepcopy(self.backbone)
        update_drop_path_rate(
            self.student_backbone.vit, drop_path_rate=0.1, mode="uniform"
        )
        self.student_dino_head = DINOProjectionHead(
            input_dim=384, freeze_last_layer=1, norm_last_layer=False
        )
        self.student_ibot_head = self.student_dino_head
        if ibot_separate_head:
            self.student_ibot_head = DINOProjectionHead(
                input_dim=384, freeze_last_layer=1, norm_last_layer=False
            )

        # Losses
        self.dino_criterion = DINOLoss()
        self.ibot_criterion = IBOTPatchLoss(output_dim=65536)
        self.koleo_criterion = KoLeoLoss()

        self.online_classifier = OnlineLinearClassifier(
            feature_dim=384, num_classes=num_classes
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def forward_teacher(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        features = self.backbone.encode(x)
        cls_tokens = features[:, 0]
        return cls_tokens, features

    def forward_student(
        self, x: Tensor, mask: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        features = self.student_backbone.encode(x, mask=mask)
        cls_tokens = features[:, 0]
        masked_features = None if mask is None else features[mask]
        return cls_tokens, masked_features

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        # Momentum update teacher.
        momentum = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=0.992,
            end_value=1.0,
        )
        update_momentum(self.student_backbone, self.backbone, m=momentum)
        update_momentum(self.student_dino_head, self.dino_head, m=momentum)

        views, targets = batch[0], batch[1]
        global_views = torch.cat(views[:2])
        local_views = torch.cat(views[2:])

        # Masking
        B = len(global_views)
        sequence_length = self.backbone.sequence_length
        num_patches = int((sequence_length - 1) ** 0.5)
        mask = global_views.new_zeros((B, sequence_length), dtype=torch.bool)
        # Mask patches except class token.
        block_mask = random_block_mask(
            size=(B, num_patches, num_patches), device=mask.device
        )
        mask[:, 1:] = block_mask.flatten(start_dim=1)

        # Teacher forward
        with torch.no_grad():
            teacher_cls_token, teacher_features = self.forward_teacher(global_views)
            teacher_cls_out = self.dino_head(teacher_cls_token)
            teacher_masked_out = self.ibot_head(teacher_features[mask])

        # Student forward
        student_global_cls_token, student_global_masked_features = self.forward_student(
            global_views, mask=mask
        )
        student_global_cls_out = self.student_dino_head(student_global_cls_token)
        student_global_masked_out = self.student_ibot_head(
            student_global_masked_features
        )

        student_local_cls_token, _ = self.forward_student(local_views, mask=None)
        student_local_cls_out = self.student_dino_head(student_local_cls_token)
        student_cls_out = torch.cat([student_global_cls_out, student_local_cls_out])

        # Loss
        teacher_temp = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=0.04,
            end_value=0.07,
        )

        # Remove teacher_temp from the loss functions
        dino_loss = self.dino_criterion(
            teacher_out=teacher_cls_out.chunk(2),
            student_out=student_cls_out.chunk(len(views)),
            epoch=self.current_epoch,
        )
        ibot_loss = self.ibot_criterion(
            teacher_out=teacher_masked_out,
            student_out=student_global_masked_out,
            mask=block_mask,
        )
        koleo_loss = 0.1 * sum(
            self.koleo_criterion(t) for t in student_global_cls_token.chunk(2)
        )
        loss = dino_loss + ibot_loss + koleo_loss

        self.log_dict(
            {
                "train_loss": loss,
                "train_dino_loss": dino_loss,
                "train_ibot_loss": ibot_loss,
                "train_koleo_loss": koleo_loss,
                "ema_momentum": momentum,
                "teacher_temp": teacher_temp,
            },
            prog_bar=True,
            sync_dist=True,
            batch_size=len(targets),
        )

        # Online classification.
        cls_loss, cls_log = self.online_classifier.training_step(
            (teacher_cls_token.chunk(2)[0].detach(), targets), batch_idx
        )
        self.log_dict(cls_log, sync_dist=True, batch_size=len(targets))
        return loss + cls_loss


    def validation_step(
        self, batch: Tuple[Tensor, Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        images, targets = batch[0], batch[1]
        cls_token = self.forward(images)
        cls_loss, cls_log = self.online_classifier.validation_step(
            (cls_token.detach(), targets), batch_idx
        )
        self.log_dict(cls_log, prog_bar=True, sync_dist=True, batch_size=len(targets))
        return cls_loss

    def configure_optimizers(self):
        lr_scale = math.sqrt(
            self.batch_size_per_device * self.trainer.world_size / 1024
        )
        lr = 0.004 * lr_scale
        weight_decay = 0.04

        lr_layer_decay = 0.9
        num_layers = len(self.student_backbone.vit.blocks)

        def lr_layer(layer_idx: int) -> float:
            return lr_layer_decay ** (num_layers + 1 - layer_idx)

        param_groups = []
        for name, param in self.named_parameters():
            if not "student" in name:
                continue  # Ignore teacher parameters

            group = {
                "name": name,
                "params": [param],
                "lr": lr,
                "weight_decay": weight_decay,
            }

            # Update lr
            if any(
                s in name
                for s in [
                    "pos_embed",
                    "mask_token",
                    "cls_token",
                    "register_tokens",
                ]
            ):
                group["lr"] = lr * lr_layer(0)
            elif "patch_embed" in name:
                group["lr"] = lr * lr_layer(0) * 0.2
            elif "residual" in name:
                group["lr"] = lr
            elif "blocks" in name:
                layer_idx = int(re.search(r"blocks\.(\d+)\.", name).group(1))
                group["lr"] = lr * lr_layer(layer_idx + 1)
            elif "vit.norm" in name:
                pass  # Do not update vit.norm parameters
            elif "head" in name:
                pass  # Do not update classification and dino/ibot head parameters
            else:
                assert False, f"Unknown parameter: {name}"

            # Update weight_decay
            if name.endswith(".bias") or ".norm" in name or "gamma" in name:
                group["weight_decay"] = 0.0

            # Ignore ViT classification head
            if not "vit.head" in name:
                param_groups.append(group)

        param_groups.append(
            {
                "name": "online_classifier",
                "params": self.online_classifier.parameters(),
                "lr": lr,
                "weight_decay": 0.0,
            }
        )
        optimizer = AdamW(param_groups, lr=lr)
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=int(
                    self.trainer.estimated_stepping_batches
                    / self.trainer.max_epochs
                    * 10
                ),
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]

# Define transform
transform = DINOTransform(
    global_crop_scale=(0.32, 1),
    local_crop_scale=(0.05, 0.32),
    n_local_views=8,
)

# Dataset setup for CIFAR-10
train_dataset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)

val_dataset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

# Model and training setup
model = DINOv2(batch_size_per_device=8, num_classes=len(train_dataset.classes))

# Training the model with PyTorch Lightning
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='./checkpoints',
    filename='dino_model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min',
)

trainer = Trainer(
    devices=4,  # Change this based on available GPUs
    max_epochs=10,
    callbacks=[checkpoint_callback],
)

trainer.fit(model, train_dataloader)
