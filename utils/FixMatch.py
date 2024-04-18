from torchvision import transforms
from utils.randaugment import rand_augment_transform
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC


rgb_mean = (0.48145466, 0.4578275, 0.40821073)
ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),)

class TransformFixMatch(object):
    def __init__(self,weak,args):
        size = 224 if args.datasets == "visda" else 256
        self.strong = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize([size, size]),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        self.weak = weak

    def __call__(self, x):
        return self.weak(x),self.strong(x)