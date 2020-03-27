import json
import os
import time
import torch
import visdom


class Logger(object):
    def __init__(self, name, output_path, is_visdom, port, save=True, session=None):
        self.save = save
        self.viz = None
        if is_visdom:
            self.viz = visdom.Visdom(port=port)
            assert self.viz.check_connection()
        start_time = time.strftime('%Y%m%d-%H-%M-%S', time.localtime(time.time()))
        if session:
            self.session = session
        else:
            self.session = "{}_{}".format(name, start_time)
        self.output_path = os.path.join(output_path, self.session)
        self.model_path = os.path.join(self.output_path, 'models')
        self.log_path = os.path.join(self.output_path, "log.txt")
        self.cfg_path = os.path.join(self.output_path, 'config.json')
        if save:
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
        self.plot_data = dict()

    def line(self, name, val, index, win=None, **kwargs):
        # if isinstance(val, float) or isinstance(val, int):
        #     val = [val]
        # if isinstance(index, float) or isinstance(index, int):
        #     index = [index]
        if self.viz:
            if win is None:
                win = name
            if win not in self.plot_data:
                self.plot_data[win] = ([], [])
            self.plot_data[win][0].append(index)
            self.plot_data[win][1].append(val)
            self.viz.line(
                Y=self.plot_data[win][1], X=self.plot_data[win][0], name=name,
                win=win, env=self.session,
                **kwargs
            )

    def show_images(self, images):
        if self.viz:
            self.viz.images(
                images, env='images',
            )
    def show_graph(self, bins, title):
        if self.viz:
            self.viz.bar(
                bins, opts={"title": title}, env='estimate',
            )

    def save_model(self, model, optim, lr_scheduler, epoch, post_fix=''):
        if self.save:
            torch.save({
                'session': self.session,
                'epoch': epoch + 1,
                'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer': optim.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
            }, "{0}/epoch_{1}.pth".format(self.model_path, str(epoch) + post_fix))

    def log(self, content):
        # self.logger.info(content)
        print("[LOG][{}] {}".format(time.strftime('%Y/%m/%d %H:%M:%S'), content))
        if self.save:
            with open(self.log_path, 'a+') as f:
                f.writelines(content + '\n')

    def log_config(self, cfg):
        cfg_str = json.dumps(cfg, indent=4)
        print("[LOG] {}".format(cfg_str))
        if self.save:
            with open(self.cfg_path, 'a+') as f:
                f.writelines(cfg_str)
                f.write(self.output_path + '\n')
