| display_name | experiment | initialization | trainable_scope | lr | epochs_run | best_epoch | best_val_acc | test_accuracy | macro_f1 | roc_auc | trainable_parameters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frozen head | resnet18_pretrained_frozen_128_e5 | ImageNet | classifier head | 0.001 | 5 | 5 | 0.8563 | 0.8697 | 0.8697 | 0.9344 | 1026 |
| Layer4 fine-tune | resnet18_layer4_from_frozen_lr3e-5_128_e8 | Frozen checkpoint | classifier + layer4 | 3e-05 | 8 | 8 | 0.9540 | 0.9406 | 0.9406 | 0.9850 | 8394754 |
| Full fine-tune LR 1e-5 | resnet18_unfrozen_from_frozen_lr1e-5_128_e8 | Frozen checkpoint | full backbone | 1e-05 | 8 | 7 | 0.9406 | 0.9310 | 0.9310 | 0.9781 | 11177538 |
| Full fine-tune LR 3e-5 | resnet18_unfrozen_from_frozen_lr3e-5_128_e8 | Frozen checkpoint | full backbone | 3e-05 | 7 | 3 | 0.9598 | 0.9406 | 0.9406 | 0.9845 | 11177538 |
| Full fine-tune LR 1e-4 | resnet18_unfrozen_from_frozen_lr1e-4_128_e8 | Frozen checkpoint | full backbone | 0.0001 | 8 | 8 | 0.9655 | 0.9617 | 0.9617 | 0.9944 | 11177538 |
