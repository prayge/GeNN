if __name__ == '__main__':    
    import pytorch_lightning
    from logging import root
    from monai.utils import first, set_determinism
    import torch
    import matplotlib.pyplot as plt
    import os
    from init import Options
    from utils import *
    from net import *
   
    opt = Options().parse()  
    net = Net()
    root = set_task_dir(opt.task)
    log_dir = os.path.join(root, "logs")
    tb_logger = pytorch_lightning.loggers.TensorBoardLogger(
        save_dir=log_dir
    )

    root = set_task_dir(opt.task)
    train_files, val_files = train_val_split(opt.trainfolder, opt.labelfolder)
    cfg = importCfg(opt.config, opt.task)
    
    print("----------------------------------------------------------------------------------------")
    print(torch.cuda.is_available())
    print("~~~~~~~~~~~~~~~~~~~~~~~~")
    print(root)
    print("~~~~~~~~~~~~~~~~~~~~~~~~")
    print(train_files)
    print("~~~~~~~~~~~~~~~~~~~~~~~~")
    print(val_files)
    print("~~~~~~~~~~~~~~~~~~~~~~~~")
    print(cfg)
    print("~~~~~~~~~~~~~~~~~~~~~~~~")
    print(opt)
    print("----------------------------------------------------------------------------------------")

    checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(dirpath=opt.modeldir, save_top_k=2, monitor="val_loss")

    trainer = pytorch_lightning.Trainer(
        callbacks=[checkpoint_callback],
        default_root_dir=opt.modeldir,
        accelerator='gpu', 
        devices=1,
        min_epochs=10,
        max_epochs=20,
        logger=tb_logger,
        log_every_n_steps=5,
        num_sanity_val_steps=1
    )

    trainer.fit(net)

    printf(
        f"train completed, best_metric: {net.best_val_dice:.4f} "
        f"at epoch {net.best_val_epoch}")

    net.eval()
    device = torch.device("cuda:0")
    net.to(device)
    with torch.no_grad():
        for i, val_data in enumerate(net.val_dataloader()):
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(
                val_data["image"].to(device), roi_size, sw_batch_size, net
            )
            # plot the slice [:, :, 80]
            plt.figure("check", (18, 6))
            plt.subplot(1, 3, 1)
            plt.title(f"image {i}")
            plt.imshow(val_data["image"][0, 0, :, :, 80], cmap="gray")
            plt.subplot(1, 3, 2)
            plt.title(f"label {i}")
            plt.imshow(val_data["label"][0, 0, :, :, 80])
            plt.subplot(1, 3, 3)
            plt.title(f"output {i}")
            plt.imshow(torch.argmax(
                val_outputs, dim=1).detach().cpu()[0, :, :, 80])
            plt.show()