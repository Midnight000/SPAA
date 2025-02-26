[Dataset] SPAA: Stealthy Projector-based Adversarial Attacks on Deep Image Classifiers
======================================================================================

## Usage

Download and extract this zip to folder ``SPAA/data`` and follow instructions [here][3]. See [paper][1] and [supplementary][2].

## Folder Structure

    ├─prj_share
    │  ├─init
    │  ├─numbers
    │  ├─test
    │  └─train
    ├─sample
    └─setups
        ├─backpack       # a setup
        │  ├─cam
        │  │  ├─infer
        │  │  │  ├─adv
        │  │  │  │  ├─PerC-AL+CompenNet++_l1+ssim_500_24_2000
        │  │  │  │  │  └─camdE
        │  │  │  │  │      └─11
        │  │  │  │  │          ├─inception_v3
        │  │  │  │  │          ├─resnet18
        │  │  │  │  │          └─vgg16
        │  │  │  │  └─SPAA_PCNet_l1+ssim_500_24_2000
        │  │  │  │      ├─camdE
        │  │  │  │      │  ├─11
        │  │  │  │      │  │  ├─inception_v3
        │  │  │  │      │  │  ├─resnet18
        │  │  │  │      │  │  └─vgg16
        │  │  │  │      │  ├─5
        │  │  │  │      │  │  ├─inception_v3
        │  │  │  │      │  │  ├─resnet18
        │  │  │  │      │  │  └─vgg16
        │  │  │  │      │  ├─7
        │  │  │  │      │  │  ├─inception_v3
        │  │  │  │      │  │  ├─resnet18
        │  │  │  │      │  │  └─vgg16
        │  │  │  │      │  └─9
        │  │  │  │      │      ├─inception_v3
        │  │  │  │      │      ├─resnet18
        │  │  │  │      │      └─vgg16
        │  │  │  │      ├─camdE_caml2
        │  │  │  │      │  ├─11
        │  │  │  │      │  │  ├─inception_v3
        │  │  │  │      │  │  ├─resnet18
        │  │  │  │      │  │  └─vgg16
        │  │  │  │      │  ├─5
        │  │  │  │      │  │  ├─inception_v3
        │  │  │  │      │  │  ├─resnet18
        │  │  │  │      │  │  └─vgg16
        │  │  │  │      │  ├─7
        │  │  │  │      │  │  ├─inception_v3
        │  │  │  │      │  │  ├─resnet18
        │  │  │  │      │  │  └─vgg16
        │  │  │  │      │  └─9
        │  │  │  │      │      ├─inception_v3
        │  │  │  │      │      ├─resnet18
        │  │  │  │      │      └─vgg16
        │  │  │  │      └─caml2
        │  │  │  │          ├─11
        │  │  │  │          │  ├─inception_v3
        │  │  │  │          │  ├─resnet18
        │  │  │  │          │  └─vgg16
        │  │  │  │          ├─5
        │  │  │  │          │  ├─inception_v3
        │  │  │  │          │  ├─resnet18
        │  │  │  │          │  └─vgg16
        │  │  │  │          ├─7
        │  │  │  │          │  ├─inception_v3
        │  │  │  │          │  ├─resnet18
        │  │  │  │          │  └─vgg16
        │  │  │  │          └─9
        │  │  │  │              ├─inception_v3
        │  │  │  │              ├─resnet18
        │  │  │  │              └─vgg16
        │  │  │  └─test
        │  │  │      ├─PCNet_l1+ssim_500_24_2000
        │  │  │      └─PCNet_no_mask_no_rough_d_l1+ssim_500_24_2000
        │  │  └─raw
        │  │      ├─adv
        │  │      │  ├─One-pixel_DE
        │  │      │  │  └─-
        │  │      │  │      └─-
        │  │      │  │          ├─inception_v3
        │  │      │  │          ├─resnet18
        │  │      │  │          └─vgg16
        │  │      │  ├─PerC-AL+CompenNet++_l1+ssim_500_24_2000
        │  │      │  │  └─camdE
        │  │      │  │      └─11
        │  │      │  │          ├─inception_v3
        │  │      │  │          ├─resnet18
        │  │      │  │          └─vgg16
        │  │      │  └─SPAA_PCNet_l1+ssim_500_24_2000
        │  │      │      ├─camdE
        │  │      │      │  ├─11
        │  │      │      │  │  ├─inception_v3
        │  │      │      │  │  ├─resnet18
        │  │      │      │  │  └─vgg16
        │  │      │      │  ├─5
        │  │      │      │  │  ├─inception_v3
        │  │      │      │  │  ├─resnet18
        │  │      │      │  │  └─vgg16
        │  │      │      │  ├─7
        │  │      │      │  │  ├─inception_v3
        │  │      │      │  │  ├─resnet18
        │  │      │      │  │  └─vgg16
        │  │      │      │  └─9
        │  │      │      │      ├─inception_v3
        │  │      │      │      ├─resnet18
        │  │      │      │      └─vgg16
        │  │      │      ├─camdE_caml2
        │  │      │      │  ├─11
        │  │      │      │  │  ├─inception_v3
        │  │      │      │  │  ├─resnet18
        │  │      │      │  │  └─vgg16
        │  │      │      │  ├─5
        │  │      │      │  │  ├─inception_v3
        │  │      │      │  │  ├─resnet18
        │  │      │      │  │  └─vgg16
        │  │      │      │  ├─7
        │  │      │      │  │  ├─inception_v3
        │  │      │      │  │  ├─resnet18
        │  │      │      │  │  └─vgg16
        │  │      │      │  └─9
        │  │      │      │      ├─inception_v3
        │  │      │      │      ├─resnet18
        │  │      │      │      └─vgg16
        │  │      │      └─caml2
        │  │      │          ├─11
        │  │      │          │  ├─inception_v3
        │  │      │          │  ├─resnet18
        │  │      │          │  └─vgg16
        │  │      │          ├─5
        │  │      │          │  ├─inception_v3
        │  │      │          │  ├─resnet18
        │  │      │          │  └─vgg16
        │  │      │          ├─7
        │  │      │          │  ├─inception_v3
        │  │      │          │  ├─resnet18
        │  │      │          │  └─vgg16
        │  │      │          └─9
        │  │      │              ├─inception_v3
        │  │      │              ├─resnet18
        │  │      │              └─vgg16
        │  │      ├─cb
        │  │      ├─ref
        │  │      ├─test
        │  │      └─train
        │  ├─prj
        │  │  ├─adv
        │  │  │  ├─One-pixel_DE
        │  │  │  │  └─-
        │  │  │  │      └─-
        │  │  │  │          ├─inception_v3
        │  │  │  │          ├─resnet18
        │  │  │  │          └─vgg16
        │  │  │  ├─PerC-AL+CompenNet++_l1+ssim_500_24_2000
        │  │  │  │  └─camdE
        │  │  │  │      └─11
        │  │  │  │          ├─inception_v3
        │  │  │  │          ├─resnet18
        │  │  │  │          └─vgg16
        │  │  │  └─SPAA_PCNet_l1+ssim_500_24_2000
        │  │  │      ├─camdE
        │  │  │      │  ├─11
        │  │  │      │  │  ├─inception_v3
        │  │  │      │  │  ├─resnet18
        │  │  │      │  │  └─vgg16
        │  │  │      │  ├─5
        │  │  │      │  │  ├─inception_v3
        │  │  │      │  │  ├─resnet18
        │  │  │      │  │  └─vgg16
        │  │  │      │  ├─7
        │  │  │      │  │  ├─inception_v3
        │  │  │      │  │  ├─resnet18
        │  │  │      │  │  └─vgg16
        │  │  │      │  └─9
        │  │  │      │      ├─inception_v3
        │  │  │      │      ├─resnet18
        │  │  │      │      └─vgg16
        │  │  │      ├─camdE_caml2
        │  │  │      │  ├─11
        │  │  │      │  │  ├─inception_v3
        │  │  │      │  │  ├─resnet18
        │  │  │      │  │  └─vgg16
        │  │  │      │  ├─5
        │  │  │      │  │  ├─inception_v3
        │  │  │      │  │  ├─resnet18
        │  │  │      │  │  └─vgg16
        │  │  │      │  ├─7
        │  │  │      │  │  ├─inception_v3
        │  │  │      │  │  ├─resnet18
        │  │  │      │  │  └─vgg16
        │  │  │      │  └─9
        │  │  │      │      ├─inception_v3
        │  │  │      │      ├─resnet18
        │  │  │      │      └─vgg16
        │  │  │      └─caml2
        │  │  │          ├─11
        │  │  │          │  ├─inception_v3
        │  │  │          │  ├─resnet18
        │  │  │          │  └─vgg16
        │  │  │          ├─5
        │  │  │          │  ├─inception_v3
        │  │  │          │  ├─resnet18
        │  │  │          │  └─vgg16
        │  │  │          ├─7
        │  │  │          │  ├─inception_v3
        │  │  │          │  ├─resnet18
        │  │  │          │  └─vgg16
        │  │  │          └─9
        │  │  │              ├─inception_v3
        │  │  │              ├─resnet18
        │  │  │              └─vgg16
        │  │  ├─infer
        │  │  │  └─test
        │  │  │      └─CompenNet++_l1+ssim_500_24_2000
        │  │  └─raw
        │  │      ├─cb
        │  │      └─ref
        │  └─ret
        │      ├─One-pixel_DE
        │      │  └─-
        │      │      └─-
        │      │          ├─inception_v3
        │      │          ├─resnet18
        │      │          └─vgg16
        │      ├─PerC-AL+CompenNet++_l1+ssim_500_24_2000
        │      │  └─camdE
        │      │      └─11
        │      │          ├─inception_v3
        │      │          ├─resnet18
        │      │          └─vgg16
        │      └─SPAA_PCNet_l1+ssim_500_24_2000
        │          ├─camdE
        │          │  ├─11
        │          │  │  ├─inception_v3
        │          │  │  ├─resnet18
        │          │  │  └─vgg16
        │          │  ├─5
        │          │  │  ├─inception_v3
        │          │  │  ├─resnet18
        │          │  │  └─vgg16
        │          │  ├─7
        │          │  │  ├─inception_v3
        │          │  │  ├─resnet18
        │          │  │  └─vgg16
        │          │  └─9
        │          │      ├─inception_v3
        │          │      ├─resnet18
        │          │      └─vgg16
        │          ├─camdE_caml2
        │          │  ├─11
        │          │  │  ├─inception_v3
        │          │  │  ├─resnet18
        │          │  │  └─vgg16
        │          │  ├─5
        │          │  │  ├─inception_v3
        │          │  │  ├─resnet18
        │          │  │  └─vgg16
        │          │  ├─7
        │          │  │  ├─inception_v3
        │          │  │  ├─resnet18
        │          │  │  └─vgg16
        │          │  └─9
        │          │      ├─inception_v3
        │          │      ├─resnet18
        │          │      └─vgg16
        │          └─caml2
        │              ├─11
        │              │  ├─inception_v3
        │              │  ├─resnet18
        │              │  └─vgg16
        │              ├─5
        │              │  ├─inception_v3
        │              │  ├─resnet18
        │              │  └─vgg16
        │              ├─7
        │              │  ├─inception_v3
        │              │  ├─resnet18
        │              │  └─vgg16
        │              └─9
        │                  ├─inception_v3
        │                  ├─resnet18
        │                  └─vgg16
        ├─banana     # another setup        

## Citation

    @inproceedings{huang2022spaa,
        title      = {SPAA: Stealthy Projector-based Adversarial Attacks on Deep Image Classifiers},
        booktitle  = {2022 IEEE Conference on Virtual Reality and 3D User Interfaces (VR)},
        author     = {Huang, Bingyao and Ling, Haibin},
        year       = {2022},
        month      = mar,
        pages      = {534--542},
        publisher  = {IEEE},
        address    = {Christchurch, New Zealand},
        doi        = {10.1109/VR51125.2022.00073},
        isbn       = {978-1-66549-617-9}
    }

## Acknowledgments

We thank the anonymous reviewers for valuable and inspiring comments and suggestions.
We thank the authors of the colorful textured sampling images.

[1]: https://bingyaohuang.github.io/pub/SPAA
[2]: https://bingyaohuang.github.io/pub/SPAA/supp
[3]: https://github.com/BingyaoHuang/SPAA

