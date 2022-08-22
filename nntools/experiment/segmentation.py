import torch
from torchmetrics import CohenKappa, JaccardIndex, Dice, BinnedPrecisionRecallCurve
from nntools.experiment.supervised_experiment import SupervisedExperiment
import segmentation_models_pytorch as smp
import torchmetrics.functional as Fmetric

class AUCPrecisionRecallCurve(BinnedPrecisionRecallCurve):
    def compute(self):
        precision, recall, thresholds = super(AUCPrecisionRecallCurve, self).compute()
        
        if isinstance(precision, list):
            out = {}
            for i, (p, r) in enumerate(zip(precision, recall)):
                out[f'PrRecAUC_Class_{i}'] = Fmetric.auc(p, r, reorder=True)
            return out

        else:
            return Fmetric.auc(precision, recall, reorder=True)
        


class SegmentationExperiment(SupervisedExperiment):
    def __init__(self, config, run_id=None, trial=None, 
                 multilabel=False,
                 ignore_score_index=0):
        super().__init__(config, run_id, trial, multilabel=multilabel)
        
        
        self.gt_name = 'mask'
        self.data_keys = ['image']
        
        self.ignore_score_index = ignore_score_index
        self.set_optimizer(**self.c['Optimizer'])

    
    def init_model(self):
        
        model_setup = self.c['Network'].copy()
        model_name = model_setup.pop('architecture')
        model_setup.pop('synchronized_batch_norm', None)
        n_classes = model_setup.pop('n_classes', None)
        model = smp.create_model(model_name, classes=n_classes, **model_setup)
        self.set_model(model)
        self.model.add_metric({'CohenKappa':CohenKappa(self.n_classes),
                                'Dice':Dice(self.n_classes, ignore_index=self.ignore_score_index),
                                'class_score':AUCPrecisionRecallCurve(self.n_classes)})
    
    def validate(self, model, valid_loader, loss_function=None):
        with torch.no_grad():
            for batch in valid_loader:
                batch = self.batch_to_device(batch, self.ctx.rank)
                preds = model(*self.pass_data_keys_to_model(batch=batch))
                preds = self.head_activation(preds)
                if self.multilabel:
                    preds = preds > 0.5
                else:
                    preds = torch.argmax(preds, 1, keepdim=True)
                break
        
        self.visualization_images(batch['image'], batch[self.gt_name], 'input_images')
        self.visualization_images(batch['image'], preds, 'output_images')        
        return super().validate(model, valid_loader, loss_function)