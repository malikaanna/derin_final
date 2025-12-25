PS C:\Users\admin\Documents\Cursor Projects\derin_final> python train_vae.py --epochs 100 --batch_size 64 --lr 0.0002
Using device: cuda
Loading dataset...
Loading data from: C:\Users\admin\Documents\Cursor Projects\derin_final\data\dataset\dataset\train
Found 10901 images
Classes: ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges']
Train size: 9811, Validation size: 1090
Creating VAE with latent_dim=128...
Total parameters: 2,958,659

Starting training for 100 epochs...
KL weight (beta): 1.0
--------------------------------------------------
Epoch [1/100] Train: 3133.5093 (Recon: 2967.1886, KL: 166.3207) Val: 1641.8122
Saved generated images to outputs\vae\samples_epoch_001.png
Epoch [2/100] Train: 1354.9409 (Recon: 1170.0736, KL: 184.8673) Val: 1162.4891
ges']
Train size: 9811, Validation size: 1090
Creating VAE with latent_dim=128...
Total parameters: 2,958,659

Starting training for 100 epochs...
KL weight (beta): 1.0
--------------------------------------------------
Epoch [1/100] Train: 3133.5093 (Recon: 2967.1886, KL: 166.3207) Val: 1641.8122
Saved generated images to outputs\vae\samples_epoch_001.png
Epoch [2/100] Train: 1354.9409 (Recon: 1170.0736, KL: 184.8673) Val: 1162.4891
Epoch [3/100] Train: 1076.8003 (Recon: 888.7935, KL: 188.0068) Val: 1067.7516
Epoch [4/100] Train: 953.7133 (Recon: 763.9146, KL: 189.7987) Val: 888.7884
Epoch [5/100] Train: 879.1262 (Recon: 689.7107, KL: 189.4155) Val: 852.1686
Saved generated images to outputs\vae\samples_epoch_005.png
Epoch [6/100] Train: 830.3985 (Recon: 641.8449, KL: 188.5535) Val: 825.9027
Epoch [7/100] Train: 798.7899 (Recon: 610.6995, KL: 188.0904) Val: 766.5353
Epoch [8/100] Train: 767.9997 (Recon: 580.9983, KL: 187.0014) Val: 755.9702
Epoch [9/100] Train: 748.4590 (Recon: 561.9919, KL: 186.4672) Val: 749.4121
Epoch [10/100] Train: 725.8613 (Recon: 540.2478, KL: 185.6135) Val: 728.2076
Saved generated images to outputs\vae\samples_epoch_010.png
Saved checkpoint: checkpoints\vae_epoch_010.pt
Epoch [11/100] Train: 707.3523 (Recon: 522.9163, KL: 184.4359) Val: 702.7441
Epoch [12/100] Train: 692.6420 (Recon: 508.9885, KL: 183.6535) Val: 696.5788
Epoch [13/100] Train: 682.1256 (Recon: 499.1032, KL: 183.0224) Val: 685.2385
Epoch [14/100] Train: 669.8894 (Recon: 487.7104, KL: 182.1790) Val: 663.3353
Epoch [15/100] Train: 661.4080 (Recon: 479.5754, KL: 181.8326) Val: 647.6793
Saved generated images to outputs\vae\samples_epoch_015.png
Epoch [16/100] Train: 648.1458 (Recon: 467.6159, KL: 180.5299) Val: 646.3567
Epoch [17/100] Train: 638.1251 (Recon: 458.6855, KL: 179.4396) Val: 647.6878
Epoch [18/100] Train: 632.9034 (Recon: 453.7855, KL: 179.1179) Val: 639.0236
Epoch [19/100] Train: 622.1969 (Recon: 444.2122, KL: 177.9847) Val: 634.2729
Epoch [20/100] Train: 616.7114 (Recon: 439.1536, KL: 177.5578) Val: 633.7290
Saved generated images to outputs\vae\samples_epoch_020.png
Saved checkpoint: checkpoints\vae_epoch_020.pt
Epoch [21/100] Train: 609.8280 (Recon: 432.7857, KL: 177.0423) Val: 610.8462
Epoch [22/100] Train: 599.6458 (Recon: 423.3466, KL: 176.2992) Val: 605.8429
Epoch [23/100] Train: 600.3532 (Recon: 424.6358, KL: 175.7174) Val: 597.2408
Epoch [24/100] Train: 594.1620 (Recon: 419.1088, KL: 175.0532) Val: 587.9933
Epoch [25/100] Train: 587.2390 (Recon: 412.3047, KL: 174.9343) Val: 620.5251
Saved generated images to outputs\vae\samples_epoch_025.png
Epoch [26/100] Train: 583.3612 (Recon: 409.2888, KL: 174.0724) Val: 585.1334
Epoch [27/100] Train: 578.3295 (Recon: 404.9168, KL: 173.4127) Val: 588.6620
Epoch [28/100] Train: 574.1508 (Recon: 401.1791, KL: 172.9717) Val: 590.1551
Epoch [29/100] Train: 571.0129 (Recon: 398.1026, KL: 172.9103) Val: 567.4812
Epoch [30/100] Train: 562.9343 (Recon: 391.2715, KL: 171.6627) Val: 565.0941
Saved generated images to outputs\vae\samples_epoch_030.png
Saved checkpoint: checkpoints\vae_epoch_030.pt
Epoch [31/100] Train: 558.7194 (Recon: 387.2601, KL: 171.4592) Val: 568.3717
Epoch [32/100] Train: 559.2955 (Recon: 388.1583, KL: 171.1372) Val: 551.0182
Epoch [33/100] Train: 554.5460 (Recon: 383.8137, KL: 170.7324) Val: 556.1911
Epoch [34/100] Train: 548.8759 (Recon: 378.8609, KL: 170.0150) Val: 561.2839
Epoch [35/100] Train: 546.9185 (Recon: 377.0571, KL: 169.8613) Val: 545.6377
Saved generated images to outputs\vae\samples_epoch_035.png
Epoch [36/100] Train: 545.2716 (Recon: 376.0928, KL: 169.1788) Val: 546.3565
Epoch [37/100] Train: 542.4682 (Recon: 373.5121, KL: 168.9561) Val: 553.1748
Epoch [38/100] Train: 538.7537 (Recon: 370.2247, KL: 168.5290) Val: 540.1950
Epoch [39/100] Train: 535.3435 (Recon: 367.3740, KL: 167.9695) Val: 534.5927
Epoch [40/100] Train: 532.5846 (Recon: 365.1317, KL: 167.4529) Val: 542.5361
Saved generated images to outputs\vae\samples_epoch_040.png
Saved checkpoint: checkpoints\vae_epoch_040.pt
Epoch [41/100] Train: 530.8352 (Recon: 363.5247, KL: 167.3105) Val: 537.2355
Epoch [42/100] Train: 529.5810 (Recon: 362.2728, KL: 167.3082) Val: 539.5290
Epoch [43/100] Train: 523.7053 (Recon: 356.8690, KL: 166.8363) Val: 534.2515
Epoch [44/100] Train: 523.7675 (Recon: 357.5118, KL: 166.2557) Val: 531.6295
Epoch [45/100] Train: 520.9850 (Recon: 354.9160, KL: 166.0690) Val: 526.7165
Saved generated images to outputs\vae\samples_epoch_045.png
Epoch [46/100] Train: 520.5380 (Recon: 354.5253, KL: 166.0127) Val: 528.2886
Epoch [47/100] Train: 515.9174 (Recon: 350.4918, KL: 165.4256) Val: 519.2168
Epoch [48/100] Train: 517.3555 (Recon: 351.9994, KL: 165.3561) Val: 523.1261
Epoch [49/100] Train: 513.4536 (Recon: 348.0620, KL: 165.3916) Val: 513.2180
Epoch [50/100] Train: 511.6643 (Recon: 347.0153, KL: 164.6490) Val: 512.6963
Saved generated images to outputs\vae\samples_epoch_050.png
Saved checkpoint: checkpoints\vae_epoch_050.pt
Epoch [51/100] Train: 510.0094 (Recon: 345.3036, KL: 164.7058) Val: 519.1522                                                                            
Epoch [52/100] Train: 510.4227 (Recon: 345.9894, KL: 164.4332) Val: 529.2800
Epoch [53/100] Train: 505.6209 (Recon: 341.4296, KL: 164.1912) Val: 521.3338
Epoch [54/100] Train: 503.7051 (Recon: 339.8094, KL: 163.8957) Val: 513.3100
Epoch [55/100] Train: 501.0522 (Recon: 337.4607, KL: 163.5915) Val: 514.9596
Saved generated images to outputs\vae\samples_epoch_055.png
Epoch [56/100] Train: 500.7611 (Recon: 337.6144, KL: 163.1467) Val: 517.6016
Epoch [57/100] Train: 498.9233 (Recon: 335.9081, KL: 163.0151) Val: 502.7426
Epoch [58/100] Train: 497.4587 (Recon: 334.5702, KL: 162.8885) Val: 516.4461
Epoch [59/100] Train: 495.1366 (Recon: 332.8831, KL: 162.2535) Val: 500.0364
Epoch [60/100] Train: 493.6372 (Recon: 331.3180, KL: 162.3192) Val: 492.7194
Saved generated images to outputs\vae\samples_epoch_060.png
Saved checkpoint: checkpoints\vae_epoch_060.pt
Epoch [61/100] Train: 491.2193 (Recon: 329.4101, KL: 161.8092) Val: 503.9577
Epoch [62/100] Train: 490.5030 (Recon: 328.4300, KL: 162.0729) Val: 496.5281
Epoch [63/100] Train: 491.7902 (Recon: 329.9338, KL: 161.8564) Val: 513.3419
Epoch [64/100] Train: 490.1207 (Recon: 328.2915, KL: 161.8292) Val: 498.9850
Epoch [65/100] Train: 486.5250 (Recon: 325.3029, KL: 161.2221) Val: 498.5729
Saved generated images to outputs\vae\samples_epoch_065.png
Epoch [66/100] Train: 488.5993 (Recon: 327.2725, KL: 161.3268) Val: 492.9011
Epoch [67/100] Train: 484.1357 (Recon: 323.1423, KL: 160.9934) Val: 490.7961
Epoch [68/100] Train: 482.3884 (Recon: 321.8592, KL: 160.5292) Val: 492.6842
Epoch [69/100] Train: 483.9593 (Recon: 323.3658, KL: 160.5935) Val: 496.2001
Epoch [70/100] Train: 482.1714 (Recon: 321.4401, KL: 160.7313) Val: 495.7635
Saved generated images to outputs\vae\samples_epoch_070.png
Saved checkpoint: checkpoints\vae_epoch_070.pt
Epoch [71/100] Train: 479.5517 (Recon: 319.2590, KL: 160.2926) Val: 484.8986
Epoch [72/100] Train: 479.8943 (Recon: 319.6673, KL: 160.2270) Val: 481.3993
Epoch [73/100] Train: 477.8926 (Recon: 317.8095, KL: 160.0831) Val: 490.6598
Epoch [74/100] Train: 477.0384 (Recon: 316.9311, KL: 160.1073) Val: 491.8466
Epoch [75/100] Train: 476.4290 (Recon: 316.6459, KL: 159.7831) Val: 489.0647
Saved generated images to outputs\vae\samples_epoch_075.png
Epoch [76/100] Train: 473.5180 (Recon: 313.9247, KL: 159.5933) Val: 479.1454
Epoch [77/100] Train: 471.4935 (Recon: 312.3003, KL: 159.1932) Val: 482.0714
Epoch [78/100] Train: 472.6710 (Recon: 313.4469, KL: 159.2242) Val: 486.6461
Epoch [79/100] Train: 472.0269 (Recon: 313.1079, KL: 158.9190) Val: 479.3449
Epoch [80/100] Train: 470.8578 (Recon: 311.7027, KL: 159.1551) Val: 473.2651
Saved generated images to outputs\vae\samples_epoch_080.png
Saved checkpoint: checkpoints\vae_epoch_080.pt
Epoch [81/100] Train: 469.6215 (Recon: 310.8696, KL: 158.7519) Val: 480.0713
Epoch [82/100] Train: 468.4275 (Recon: 309.8250, KL: 158.6025) Val: 473.0384
Epoch [83/100] Train: 468.7284 (Recon: 310.0796, KL: 158.6487) Val: 468.2445
Epoch [84/100] Train: 466.5900 (Recon: 308.3116, KL: 158.2783) Val: 470.1192
Epoch [85/100] Train: 464.5890 (Recon: 306.3976, KL: 158.1913) Val: 468.2463
Saved generated images to outputs\vae\samples_epoch_085.png
Epoch [86/100] Train: 466.6403 (Recon: 308.5734, KL: 158.0669) Val: 473.5209
Epoch [87/100] Train: 463.9060 (Recon: 306.0181, KL: 157.8879) Val: 474.1183
Epoch [88/100] Train: 463.3723 (Recon: 305.4849, KL: 157.8874) Val: 468.7286
Epoch [89/100] Train: 462.4169 (Recon: 304.6181, KL: 157.7987) Val: 470.8006
Epoch [90/100] Train: 461.7882 (Recon: 304.1037, KL: 157.6846) Val: 469.7751
Saved generated images to outputs\vae\samples_epoch_090.png
Saved checkpoint: checkpoints\vae_epoch_090.pt
Epoch [91/100] Train: 460.2463 (Recon: 302.6938, KL: 157.5525) Val: 466.2200
Epoch [92/100] Train: 460.2437 (Recon: 302.9044, KL: 157.3393) Val: 466.6916
Epoch [93/100] Train: 459.3300 (Recon: 302.0932, KL: 157.2368) Val: 476.8803
Epoch [94/100] Train: 458.4725 (Recon: 301.3530, KL: 157.1196) Val: 467.5463
Epoch [95/100] Train: 458.3667 (Recon: 301.2097, KL: 157.1570) Val: 461.6215
Saved generated images to outputs\vae\samples_epoch_095.png
Epoch [96/100] Train: 455.0788 (Recon: 298.2621, KL: 156.8167) Val: 468.6720
Epoch [97/100] Train: 457.2022 (Recon: 300.5729, KL: 156.6293) Val: 466.1624
Epoch [98/100] Train: 455.7930 (Recon: 298.9675, KL: 156.8256) Val: 475.6676
Epoch [99/100] Train: 454.9704 (Recon: 298.2068, KL: 156.7636) Val: 461.6365
Epoch [100/100] Train: 453.5964 (Recon: 297.0247, KL: 156.5717) Val: 463.5106
Saved generated images to outputs\vae\samples_epoch_100.png
Saved checkpoint: checkpoints\vae_epoch_100.pt

Saved final model: checkpoints\vae_final.pt
Saved loss plot to outputs\vae\training_losses.png

Generating final samples...
Saved generated images to outputs\vae\final_samples.png

==================================================
Training completed!
Best validation loss: 461.6215
Outputs saved to: outputs\vae
Checkpoints saved to: checkpoints





































PS C:\Users\admin\Documents\Cursor Projects\derin_final> python train_dcgan.py --epochs 100 --batch_size 64 --lr 0.0002
Using device: cuda
Loading dataset...
Loading data from: C:\Users\admin\Documents\Cursor Projects\derin_final\data\dataset\dataset\train
Found 10901 images
Classes: ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges']
Train size: 9811, Validation size: 1090
Creating DCGAN with noise_dim=100, ngf=64, ndf=64...
Generator parameters: 3,576,704
Discriminator parameters: 2,765,568
Total parameters: 6,342,272

Starting training for 100 epochs...
Label smoothing: 0.1
--------------------------------------------------
Epoch [1/100] D_loss: 0.8250 G_loss: 12.5748 D(x): 0.811 D(G(z)): 0.001
Saved generated images to outputs\dcgan\samples_epoch_001.png
Epoch [2/100] D_loss: 0.6215 G_loss: 4.7451 D(x): 0.807 D(G(z)): 0.024
Epoch [3/100] D_loss: 0.6226 G_loss: 3.9919 D(x): 0.809 D(G(z)): 0.038
Epoch [4/100] D_loss: 0.6602 G_loss: 3.6937 D(x): 0.793 D(G(z)): 0.056                                                                                  
Epoch [5/100] D_loss: 0.6155 G_loss: 3.5324 D(x): 0.804 D(G(z)): 0.058                                                                                  
Saved generated images to outputs\dcgan\samples_epoch_005.png
Epoch [6/100] D_loss: 0.6700 G_loss: 3.0611 D(x): 0.776 D(G(z)): 0.086
Epoch [7/100] D_loss: 0.6964 G_loss: 3.0037 D(x): 0.768 D(G(z)): 0.097                                                                                  
Epoch [8/100] D_loss: 0.7024 G_loss: 2.7490 D(x): 0.759 D(G(z)): 0.110                                                                                  
Epoch [9/100] D_loss: 0.7272 G_loss: 2.7005 D(x): 0.748 D(G(z)): 0.119                                                                                  
Epoch [10/100] D_loss: 0.6901 G_loss: 2.6622 D(x): 0.758 D(G(z)): 0.116                                                                                 
Saved generated images to outputs\dcgan\samples_epoch_010.png
Saved checkpoint: checkpoints\dcgan_epoch_010.pt
Epoch [11/100] D_loss: 0.7547 G_loss: 2.5507 D(x): 0.739 D(G(z)): 0.129
Epoch [12/100] D_loss: 0.6744 G_loss: 2.5945 D(x): 0.757 D(G(z)): 0.118                                                                                 
Epoch [13/100] D_loss: 0.7317 G_loss: 2.6116 D(x): 0.747 D(G(z)): 0.122                                                                                 
Epoch [14/100] D_loss: 0.6489 G_loss: 2.5745 D(x): 0.763 D(G(z)): 0.112                                                                                 
Epoch [15/100] D_loss: 0.6993 G_loss: 2.5465 D(x): 0.757 D(G(z)): 0.120                                                                                 
Saved generated images to outputs\dcgan\samples_epoch_015.png
Epoch [16/100] D_loss: 0.7522 G_loss: 2.5895 D(x): 0.747 D(G(z)): 0.129
Epoch [17/100] D_loss: 0.6313 G_loss: 2.5565 D(x): 0.764 D(G(z)): 0.106                                                                                 
Epoch [18/100] D_loss: 0.6385 G_loss: 2.6349 D(x): 0.771 D(G(z)): 0.112                                                                                 
Epoch [19/100] D_loss: 0.6706 G_loss: 2.5876 D(x): 0.761 D(G(z)): 0.120                                                                                 
Epoch [20/100] D_loss: 0.6286 G_loss: 2.6355 D(x): 0.770 D(G(z)): 0.104                                                                                 
Saved generated images to outputs\dcgan\samples_epoch_020.png
Saved checkpoint: checkpoints\dcgan_epoch_020.pt
Epoch [21/100] D_loss: 0.6966 G_loss: 2.4768 D(x): 0.759 D(G(z)): 0.124
Epoch [22/100] D_loss: 0.6277 G_loss: 2.6363 D(x): 0.773 D(G(z)): 0.110                                                                                 
Epoch [23/100] D_loss: 0.6444 G_loss: 2.5872 D(x): 0.770 D(G(z)): 0.115
Epoch [24/100] D_loss: 0.6245 G_loss: 2.6322 D(x): 0.773 D(G(z)): 0.107
Epoch [25/100] D_loss: 0.6603 G_loss: 2.7089 D(x): 0.765 D(G(z)): 0.110
Saved generated images to outputs\dcgan\samples_epoch_025.png
Epoch [26/100] D_loss: 0.7867 G_loss: 2.5814 D(x): 0.745 D(G(z)): 0.131
Epoch [27/100] D_loss: 0.5986 G_loss: 2.4761 D(x): 0.774 D(G(z)): 0.114
Epoch [28/100] D_loss: 0.5953 G_loss: 2.6365 D(x): 0.779 D(G(z)): 0.105
Epoch [29/100] D_loss: 0.6649 G_loss: 2.5685 D(x): 0.765 D(G(z)): 0.122
Epoch [30/100] D_loss: 0.6836 G_loss: 2.6005 D(x): 0.767 D(G(z)): 0.116
Saved generated images to outputs\dcgan\samples_epoch_030.png
Saved checkpoint: checkpoints\dcgan_epoch_030.pt
Epoch [31/100] D_loss: 0.6202 G_loss: 2.5834 D(x): 0.775 D(G(z)): 0.114
Epoch [32/100] D_loss: 0.6272 G_loss: 2.5868 D(x): 0.774 D(G(z)): 0.114
Epoch [33/100] D_loss: 0.6561 G_loss: 2.6052 D(x): 0.770 D(G(z)): 0.117
Epoch [34/100] D_loss: 0.6376 G_loss: 2.6781 D(x): 0.772 D(G(z)): 0.114
Epoch [35/100] D_loss: 0.6401 G_loss: 2.6683 D(x): 0.775 D(G(z)): 0.108
Saved generated images to outputs\dcgan\samples_epoch_035.png
Epoch [36/100] D_loss: 0.6683 G_loss: 2.6687 D(x): 0.769 D(G(z)): 0.115
Epoch [37/100] D_loss: 0.5586 G_loss: 2.6823 D(x): 0.793 D(G(z)): 0.098                                                                                 
Epoch [38/100] D_loss: 0.7051 G_loss: 2.6918 D(x): 0.760 D(G(z)): 0.116
Epoch [39/100] D_loss: 0.6033 G_loss: 2.6684 D(x): 0.783 D(G(z)): 0.108
Epoch [40/100] D_loss: 0.5903 G_loss: 2.7187 D(x): 0.788 D(G(z)): 0.100
Saved generated images to outputs\dcgan\samples_epoch_040.png
Saved checkpoint: checkpoints\dcgan_epoch_040.pt
Epoch [41/100] D_loss: 0.6771 G_loss: 2.7185 D(x): 0.767 D(G(z)): 0.111
Epoch [35/100] D_loss: 0.6401 G_loss: 2.6683 D(x): 0.775 D(G(z)): 0.108
Saved generated images to outputs\dcgan\samples_epoch_035.png
Epoch [36/100] D_loss: 0.6683 G_loss: 2.6687 D(x): 0.769 D(G(z)): 0.115
Epoch [37/100] D_loss: 0.5586 G_loss: 2.6823 D(x): 0.793 D(G(z)): 0.098
Epoch [38/100] D_loss: 0.7051 G_loss: 2.6918 D(x): 0.760 D(G(z)): 0.116
Epoch [39/100] D_loss: 0.6033 G_loss: 2.6684 D(x): 0.783 D(G(z)): 0.108
Epoch [40/100] D_loss: 0.5903 G_loss: 2.7187 D(x): 0.788 D(G(z)): 0.100
Saved generated images to outputs\dcgan\samples_epoch_040.png
Saved checkpoint: checkpoints\dcgan_epoch_040.pt
Epoch [41/100] D_loss: 0.6771 G_loss: 2.7185 D(x): 0.767 D(G(z)): 0.111
Epoch [39/100] D_loss: 0.6033 G_loss: 2.6684 D(x): 0.783 D(G(z)): 0.108
Epoch [40/100] D_loss: 0.5903 G_loss: 2.7187 D(x): 0.788 D(G(z)): 0.100
Saved generated images to outputs\dcgan\samples_epoch_040.png
Saved checkpoint: checkpoints\dcgan_epoch_040.pt
Epoch [41/100] D_loss: 0.6771 G_loss: 2.7185 D(x): 0.767 D(G(z)): 0.111
Saved checkpoint: checkpoints\dcgan_epoch_040.pt
Epoch [41/100] D_loss: 0.6771 G_loss: 2.7185 D(x): 0.767 D(G(z)): 0.111
Epoch [42/100] D_loss: 0.7433 G_loss: 2.6545 D(x): 0.760 D(G(z)): 0.124
Epoch [43/100] D_loss: 0.5391 G_loss: 2.6875 D(x): 0.798 D(G(z)): 0.093
Epoch [44/100] D_loss: 0.7246 G_loss: 2.6711 D(x): 0.759 D(G(z)): 0.121
Epoch [45/100] D_loss: 0.5634 G_loss: 2.6810 D(x): 0.793 D(G(z)): 0.097
Saved generated images to outputs\dcgan\samples_epoch_045.png
Epoch [46/100] D_loss: 0.5853 G_loss: 2.7856 D(x): 0.789 D(G(z)): 0.096
Epoch [47/100] D_loss: 0.5750 G_loss: 2.8145 D(x): 0.793 D(G(z)): 0.097
Epoch [48/100] D_loss: 0.6808 G_loss: 2.6976 D(x): 0.770 D(G(z)): 0.116
Epoch [49/100] D_loss: 0.6669 G_loss: 2.7534 D(x): 0.780 D(G(z)): 0.107
Epoch [50/100] D_loss: 0.5772 G_loss: 2.6914 D(x): 0.787 D(G(z)): 0.095
Saved generated images to outputs\dcgan\samples_epoch_050.png
Saved checkpoint: checkpoints\dcgan_epoch_050.pt
Epoch [51/100] D_loss: 0.6516 G_loss: 2.7504 D(x): 0.777 D(G(z)): 0.105
Epoch [52/100] D_loss: 0.6618 G_loss: 2.6859 D(x): 0.774 D(G(z)): 0.109
Epoch [53/100] D_loss: 0.5848 G_loss: 2.7568 D(x): 0.790 D(G(z)): 0.102
Epoch [54/100] D_loss: 0.5324 G_loss: 2.8580 D(x): 0.809 D(G(z)): 0.084
Epoch [55/100] D_loss: 0.6054 G_loss: 2.8549 D(x): 0.787 D(G(z)): 0.099
Saved generated images to outputs\dcgan\samples_epoch_055.png
Epoch [56/100] D_loss: 0.5804 G_loss: 2.8598 D(x): 0.795 D(G(z)): 0.089
Epoch [57/100] D_loss: 0.6726 G_loss: 2.8052 D(x): 0.780 D(G(z)): 0.107
Epoch [58/100] D_loss: 0.6250 G_loss: 2.7143 D(x): 0.783 D(G(z)): 0.106
Epoch [59/100] D_loss: 0.5587 G_loss: 2.8337 D(x): 0.801 D(G(z)): 0.089
Epoch [60/100] D_loss: 0.7345 G_loss: 2.8101 D(x): 0.778 D(G(z)): 0.112
Saved generated images to outputs\dcgan\samples_epoch_060.png
Saved checkpoint: checkpoints\dcgan_epoch_060.pt
Epoch [61/100] D_loss: 0.5719 G_loss: 2.6648 D(x): 0.790 D(G(z)): 0.100
Epoch [62/100] D_loss: 0.4987 G_loss: 2.9435 D(x): 0.820 D(G(z)): 0.072
Epoch [63/100] D_loss: 0.6944 G_loss: 2.7739 D(x): 0.777 D(G(z)): 0.112
Epoch [64/100] D_loss: 0.5395 G_loss: 2.8589 D(x): 0.805 D(G(z)): 0.083
Epoch [65/100] D_loss: 0.5678 G_loss: 2.9192 D(x): 0.800 D(G(z)): 0.090
Saved generated images to outputs\dcgan\samples_epoch_065.png
Epoch [66/100] D_loss: 0.5766 G_loss: 2.8989 D(x): 0.796 D(G(z)): 0.091
Epoch [67/100] D_loss: 0.6652 G_loss: 2.9660 D(x): 0.795 D(G(z)): 0.093
Epoch [68/100] D_loss: 0.5926 G_loss: 2.7238 D(x): 0.790 D(G(z)): 0.101                                                                                 
Epoch [69/100] D_loss: 0.5251 G_loss: 2.9945 D(x): 0.815 D(G(z)): 0.076
Epoch [70/100] D_loss: 0.5341 G_loss: 2.9617 D(x): 0.812 D(G(z)): 0.082
Saved generated images to outputs\dcgan\samples_epoch_070.png
Saved checkpoint: checkpoints\dcgan_epoch_070.pt
Epoch [71/100] D_loss: 0.6034 G_loss: 2.9140 D(x): 0.795 D(G(z)): 0.096
Epoch [72/100] D_loss: 0.5481 G_loss: 2.9944 D(x): 0.810 D(G(z)): 0.081                                                                                 
Epoch [73/100] D_loss: 0.6479 G_loss: 2.8528 D(x): 0.782 D(G(z)): 0.105                                                                                 
Epoch [74/100] D_loss: 0.5010 G_loss: 2.9868 D(x): 0.822 D(G(z)): 0.071                                                                                 
Epoch [75/100] D_loss: 0.6795 G_loss: 2.9012 D(x): 0.786 D(G(z)): 0.103                                                                                 
Saved generated images to outputs\dcgan\samples_epoch_075.png
Epoch [76/100] D_loss: 0.5443 G_loss: 2.9476 D(x): 0.808 D(G(z)): 0.084
Epoch [77/100] D_loss: 0.5262 G_loss: 2.9808 D(x): 0.815 D(G(z)): 0.075                                                                                 
Epoch [78/100] D_loss: 0.5613 G_loss: 2.9821 D(x): 0.807 D(G(z)): 0.088                                                                                 
Epoch [79/100] D_loss: 0.5593 G_loss: 2.9817 D(x): 0.806 D(G(z)): 0.082                                                                                 
Epoch [80/100] D_loss: 0.5020 G_loss: 3.0981 D(x): 0.826 D(G(z)): 0.069                                                                                 
Saved generated images to outputs\dcgan\samples_epoch_080.png
Saved checkpoint: checkpoints\dcgan_epoch_080.pt
Epoch [81/100] D_loss: 0.5086 G_loss: 3.1296 D(x): 0.826 D(G(z)): 0.068
Epoch [82/100] D_loss: 0.7075 G_loss: 3.0396 D(x): 0.782 D(G(z)): 0.105                                                                                 
Epoch [83/100] D_loss: 0.5127 G_loss: 3.0262 D(x): 0.816 D(G(z)): 0.071                                                                                 
Epoch [84/100] D_loss: 0.4985 G_loss: 3.0680 D(x): 0.827 D(G(z)): 0.070                                                                                 
Epoch [85/100] D_loss: 0.6427 G_loss: 3.1978 D(x): 0.809 D(G(z)): 0.082                                                                                 
Saved generated images to outputs\dcgan\samples_epoch_085.png
Epoch [86/100] D_loss: 0.5695 G_loss: 2.8809 D(x): 0.798 D(G(z)): 0.088
Epoch [87/100] D_loss: 0.4930 G_loss: 3.0771 D(x): 0.830 D(G(z)): 0.067                                                                                 
Epoch [88/100] D_loss: 0.5329 G_loss: 3.1301 D(x): 0.817 D(G(z)): 0.074                                                                                 
Epoch [89/100] D_loss: 0.6299 G_loss: 3.0425 D(x): 0.801 D(G(z)): 0.089                                                                                 
Epoch [90/100] D_loss: 0.5021 G_loss: 3.1072 D(x): 0.825 D(G(z)): 0.069                                                                                 
Saved generated images to outputs\dcgan\samples_epoch_090.png
Saved checkpoint: checkpoints\dcgan_epoch_090.pt
Epoch [91/100] D_loss: 0.5287 G_loss: 3.2113 D(x): 0.822 D(G(z)): 0.069
Epoch [92/100] D_loss: 0.4868 G_loss: 3.2265 D(x): 0.832 D(G(z)): 0.061                                                                                 
Epoch [93/100] D_loss: 0.5123 G_loss: 3.1875 D(x): 0.826 D(G(z)): 0.068                                                                                 
Epoch [94/100] D_loss: 0.4659 G_loss: 3.3319 D(x): 0.843 D(G(z)): 0.055                                                                                 
Epoch [95/100] D_loss: 0.7600 G_loss: 2.8932 D(x): 0.776 D(G(z)): 0.109                                                                                 
Saved generated images to outputs\dcgan\samples_epoch_095.png
Epoch [96/100] D_loss: 0.5487 G_loss: 3.1160 D(x): 0.817 D(G(z)): 0.077
Epoch [97/100] D_loss: 0.5125 G_loss: 3.1550 D(x): 0.824 D(G(z)): 0.065                                                                                 
Epoch [98/100] D_loss: 0.6544 G_loss: 3.0669 D(x): 0.797 D(G(z)): 0.093                                                                                 
Epoch [99/100] D_loss: 0.4684 G_loss: 3.1388 D(x): 0.838 D(G(z)): 0.059                                                                                 
Epoch [100/100] D_loss: 0.4821 G_loss: 3.2467 D(x): 0.837 D(G(z)): 0.060                                                                                
Saved generated images to outputs\dcgan\samples_epoch_100.png
Saved checkpoint: checkpoints\dcgan_epoch_100.pt

Saved final model: checkpoints\dcgan_final.pt
Saved loss plot to outputs\dcgan\training_losses.png
Saved loss plot to outputs\dcgan\discriminator_outputs.png

Generating final samples...
Saved generated images to outputs\dcgan\final_samples.png

==================================================
Training completed!
Final D_loss: 0.4821
Final G_loss: 3.2467
Outputs saved to: outputs\dcgan
Checkpoints saved to: checkpoints
PS C:\Users\admin\Documents\Cursor Projects\derin_final>










































PS C:\Users\admin\Documents\Cursor Projects\derin_final> python evaluate.py
Using device: cuda

==================================================
Loading models...
==================================================
Loaded VAE from checkpoints/vae_final.pt
  Epoch: 100
Loaded Generator from checkpoints/dcgan_final.pt
  Epoch: 100

==================================================
Generating samples...
==================================================
VAE samples shape: torch.Size([64, 3, 64, 64])
GAN samples shape: torch.Size([64, 3, 64, 64])
Saved generated images to outputs\evaluation\vae_samples.png
Saved generated images to outputs\evaluation\gan_samples.png
Loading data from: C:\Users\admin\Documents\Cursor Projects\derin_final\data\dataset\dataset\train
Found 10901 images
Classes: ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges']
Train size: 9811, Validation size: 1090
Saved generated images to outputs\evaluation\real_samples.png
Real samples shape: torch.Size([64, 3, 64, 64])

Creating comparison figure...
Saved comparison: outputs\evaluation\model_comparison.png

==================================================
Evaluation Summary
==================================================
VAE samples saved to: outputs\evaluation\vae_samples.png
GAN samples saved to: outputs\evaluation\gan_samples.png
Comparison saved to: outputs\evaluation\model_comparison.png

📊 Karşılaştırma Kriterleri (Comparison Criteria):
--------------------------------------------------
1. Görsel Kalite (Visual Quality):
   - VAE: Daha bulanık ama tutarlı görüntüler
   - GAN: Daha keskin ama bazen tutarsız görüntüler

2. Çeşitlilik (Diversity):
   - VAE: Latent space'ten örnekleme sayesinde iyi çeşitlilik
   - GAN: Mode collapse riski, sınırlı çeşitlilik olabilir

3. Eğitim Kararlılığı (Training Stability):
   - VAE: Kararlı eğitim, loss sürekli azalır
   - GAN: Dengesiz eğitim, G ve D arasında denge gerekir
PS C:\Users\admin\Documents\Cursor Projects\derin_final> 































PS C:\Users\admin\Documents\Cursor Projects\derin_final> python compare_models.py
Using device: cuda

Loading models...

Generating samples...
Loading data from: C:\Users\admin\Documents\Cursor Projects\derin_final\data\dataset\dataset\train
Found 10901 images
Classes: ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges']
Train size: 9811, Validation size: 1090

Computing metrics...
Saved metrics to outputs\comparison\metrics.json

Saved report to outputs\comparison\comparison_report.txt

============================================================
GAN vs VAE KARŞILAŞTIRMA RAPORU
Comparative Analysis Report
============================================================

Tarih: 2025-12-25 03:21:49

------------------------------------------------------------
1. PİKSEL İSTATİSTİKLERİ (Pixel Statistics)
------------------------------------------------------------

VAE:
  Mean: 0.5951
  Std:  0.2776

DCGAN:
  Mean: 0.5888
  Std:  0.3527

------------------------------------------------------------
2. ÇEŞİTLİLİK SKORU (Diversity Score)
------------------------------------------------------------

VAE Diversity:   71.1807
DCGAN Diversity: 97.2981

→ DCGAN daha çeşitli görüntüler üretiyor

------------------------------------------------------------
3. ÜRETİM HIZİ (Generation Speed)
------------------------------------------------------------

VAE:
  Ortalama süre: 0.92 ms
  Görüntü/saniye: 69728.9

DCGAN:
  Ortalama süre: 1.75 ms
  Görüntü/saniye: 36632.0

------------------------------------------------------------
4. MODEL PARAMETRELERİ (Model Parameters)
------------------------------------------------------------

VAE Parameters:   2,958,659
DCGAN Parameters: 3,576,704

------------------------------------------------------------
5. GENEL DEĞERLENDİRME (Overall Assessment)
------------------------------------------------------------

VAE Güçlü Yönleri:
  ✓ Kararlı eğitim süreci
  ✓ Latent space üzerinde interpolasyon yapılabilir
  ✓ Reconstruction + Generation yapabilir
  ✓ Mode collapse problemi yok

VAE Zayıf Yönleri:
  ✗ Üretilen görüntüler bulanık olabilir
  ✗ MSE loss detayları kaybedebilir

DCGAN Güçlü Yönleri:
  ✓ Keskin ve detaylı görüntüler
  ✓ Gerçekçi dokular üretebilir
  ✓ Yüksek çözünürlüklü üretim potansiyeli

DCGAN Zayıf Yönleri:
  ✗ Eğitim dengesizlikleri (mode collapse)
  ✗ Hiperparametre hassasiyeti
  ✗ Değerlendirme zorluğu

============================================================

Saved visual comparison to outputs\comparison\visual_comparison.png
Saved generated images to outputs\comparison\vae_samples.png
Saved generated images to outputs\comparison\gan_samples.png

✅ Comparison completed!
PS C:\Users\admin\Documents\Cursor Projects\derin_final> 