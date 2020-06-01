import os

from visdom import Visdom

import torch
import torch.nn.functional as F
import pickle

from ignite.engine import Events, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Recall, Precision
from metric import Sensitivity, Specificity

from config import Flags
from models import Baseline
from adni_dataset import get_dataloader, process_ninecrop_batch, mean_over_ninecrop


def run_fold(parser, vis):
    devices = parser.args.devices
    parser.args.ckpt_dir = os.path.join(
        'checkpoint', parser.args.model, 'f'+str(parser.args.cur_fold))
    FG = parser.load()
    FG.devices = devices
    print(FG)

    torch.cuda.set_device(FG.devices[0])
    device = torch.device(FG.devices[0])

    net = Baseline(FG.ckpt_dir, len(FG.labels))

    performances = net.load(epoch=None, is_best=True)
    net = net.to(device)

    trainloader, testloader = get_dataloader(
        k=FG.fold, cur_fold=FG.cur_fold, modality=FG.modality, axis=FG.axis,
        labels=FG.labels, batch_size=FG.batch_size)

    evaluator = create_supervised_evaluator(
        net, device=device, non_blocking=True,
        prepare_batch=process_ninecrop_batch,
        metrics={'sensitivity': Recall(False, mean_over_ninecrop),
                 'precision': Precision(False, mean_over_ninecrop),
                 'specificity': Specificity(False, mean_over_ninecrop)})

    class Tracker(object):
        def __init__(self):
            self.data = []
    outputs = Tracker()
    targets = Tracker()

    @evaluator.on(Events.ITERATION_COMPLETED)
    def transform_ninecrop_output(engine):
        output, target = engine.state.output

        if output.size(0) != target.size(0):
            n = target.size(0)
            npatches = output.size(0) // n
            output = output.view(n, npatches, *output.shape[1:])
            output = torch.mean(output, dim=1)
        outputs.data += [output]
        targets.data += [target]

    evaluator.run(testloader)
    string = 'Fold {}'.format(FG.cur_fold) + '<br>'
    string += 'Epoch {}'.format(performances.pop('epoch')) + '<br>'
    for k in sorted(performances.keys()):
        string += k + ': ' + '{:.4f}'.format(performances[k])
        string += '<br>'

    string += 'pre : ' + str(evaluator.state.metrics['precision']) + '<br>'
    string += 'sen : ' + str(evaluator.state.metrics['sensitivity']) + '<br>'
    string += 'spe : ' + str(evaluator.state.metrics['specificity']) + '<br>'

    vis.text(string, win=FG.model+'_result_fold{}'.format(FG.cur_fold))

    del net
    return outputs.data, targets.data


if __name__ == '__main__':
    parser = Flags()
    parser.set_arguments()
    FG = parser.parse_args()
    vis = Visdom(port=FG.vis_port, env=FG.model+'_result')
    acc = Accuracy()
    loss = Loss(F.cross_entropy)
    precision = Precision()
    sensitivity = Sensitivity()
    specificity = Specificity()

    for i in range(FG.fold):
        parser.args.cur_fold = i
        output, target = run_fold(parser, vis)
        output = torch.cat(output)
        target = torch.cat(target)

        arg = (output, target)

        acc.update(arg)
        loss.update(arg)
        precision.update(arg)
        sensitivity.update(arg)
        specificity.update(arg)

    end = '<br>'
    text = 'Over all result<br>'
    text += 'accuracy:    ' + '{:.4f}'.format(acc.compute()) + end
    text += 'loss:        ' + '{:.4f}'.format(loss.compute()) + end
    text += 'precision:   ' + '{}'.format(precision.compute()) + end
    text += 'sensitivity: ' + '{}'.format(sensitivity.compute()) + end
    text += 'specificity: ' + '{}'.format(specificity.compute()) + end

    vis.text(text, 'result_overall')

    vis.save([vis.env])
