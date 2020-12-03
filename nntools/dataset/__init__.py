from nntools.dataset.image_tools import nntools_wrapper
from nntools.dataset.seg_dataset import SegmentationDataset
from nntools.dataset.tools import Composition, DataAugment
from nntools.dataset.utils import get_class_count, class_weighting, random_split

if __name__ == '__main__':
    from nntools.dataset.tools import Composition, DataAugment

    root_img = '/home/clement/Documents/phd/DR/MessidorAnnotation/img/images/'
    root_gt = '/home/clement/Documents/phd/DR/MessidorAnnotation/labelId/'
    dataset = SegmentationDataset(root_img, root_gt, shape=(800, 800))

    import albumentations as A

    img_path = '/home/clement/Documents/phd/DR/MessidorAnnotation/img/images/'
    gt_path = '/home/clement/Documents/phd/DR/MessidorAnnotation/labelId/'
    gt_path = None

    dataset = SegmentationDataset(img_path, gt_path, shape=(1500, 1500))

    aug = A.Compose([
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=30, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
        ], p=0.8),
        A.CLAHE(p=0.8),
        A.RandomBrightnessContrast(p=0.8),
        A.RandomGamma(p=0.8)])
    from nntools.dataset.tools import Composition, DataAugment
    from nntools.dataset.image_tools import normalize
    config = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

    composer = Composition(**config)
    composer << normalize
    dataset.set_composition(composer)
    dataset.plot(0)
