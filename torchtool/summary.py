import numpy as np
import torch


class Scalar(object):
    def __init__(self, vis, win, name, opts):
        self.vis = vis
        self.win = win
        self.name = name
        self.opts = opts
        self.data = []

    def _rectify(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().item()
        return np.array([x])

    def __call__(self, x, y, clear=False):
        x, y = self._rectify(x), self._rectify(y)
        res = self.vis.line(X=x, Y=y, win=self.win, name=self.name,
                            update='append' if not clear else 'remove',
                            opts=self.opts)
        if res == 'win does not exist':
            self.vis.line(X=x, Y=y, win=self.win,
                          name=self.name, opts=self.opts)

        self.data += [(x[0], y[0])]


class Image3D(object):
    def __init__(self, vis, win, update_every=1, nimages=1):
        self.vis = vis
        self.win = win
        self.num_call = -1
        self.update_every = update_every
        self.nimages = nimages

    def _rectify(self, image):
        assert isinstance(image, torch.Tensor)
        image = image.detach()
        if image.is_cuda:
            image = image.cpu()

        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.dim() == 4:
            image = image.unsqueeze(0)
        assert image.dim() == 5
        if image.dim() == 4:
            assert image.shape[0] == 1 or image.shape[0] == 3
        elif image.dim() == 5:
            assert image.shape[1] == 1 or image.shape[1] == 3

        return image

    def _same_size(self, images):
        shapes = [image.shape for image in images]
        pannel_shape = np.max(shapes, axis=0)
        pannels = [torch.ones(tuple(pannel_shape)) for _ in range(len(images))]
        for pannel, image in zip(pannels, images):
            s = image.shape
            pannel[:, :s[1], :s[2]] *= image

        return pannels

    def _slice_stack(self, image, nim):
        if nim == 1:
            ci = (np.array(image.shape[1:])//2)
            images = [image[:, ci[0], :, :],
                      image[:, :, ci[1], :],
                      image[:, :, :, ci[2]]]
        else:
            getters = [np.linspace(0, axis-1, nim, dtype=int).tolist()
                       for axis in image.shape[1:]]
            images = []
            for i0, i1, i2 in zip(*getters):
                images += [image[:, i0, :, :]]
                images += [image[:, :, i1, :]]
                images += [image[:, :, :, i2]]

        images = torch.stack(self._same_size(images))
        return images

    def __call__(self, name, image):
        self.num_call += 1
        if self.num_call % self.update_every != 0:
            return

        image = self._rectify(image)

        images = torch.cat([self._slice_stack(image[i, ...], self.nimages)
                            for i in range(image.shape[0])])
        self.vis.images(images, win=self.win, nrow=3,
                        opts=dict(caption=name, title=name))
