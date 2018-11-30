import numpy as np
import os
import ntpath
import time
from . import util
from . import html

class Visualizer():
    def __init__(self, port = 8097, web_dir = 'web'):
        # self.opt = opt
        self.display_id = 1
        self.use_html = True
        self.win_size = 128
        self.name = 'default'
        # self.option = opt
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port = port)
            self.display_single_pane_ncols = 0

        if self.use_html:
            self.web_dir = os.path.join('./', self.name, web_dir)
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join('./', self.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch):
        if self.display_id > 0: # show images in the browser
            if self.display_single_pane_ncols > 0:
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
    table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
    table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
</style>""" % (w, h)
                ncols = self.display_single_pane_ncols
                title = self.name
                label_html = ''
                label_html_row = ''
                nrows = int(np.ceil(len(visuals.items()) / ncols))
                images = []
                idx = 0
                for label, image_numpy in visuals.items():
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1]))*255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                padding=2, opts=dict(title=title + ' images'))
                label_html = '<table>%s</table>' % label_html
                self.vis.text(table_css + label_html, win = self.display_id + 2,
                              opts=dict(title=title + ' labels'))
            else:
                idx = 1
                for label, image_numpy in visuals.items():
                    #image_numpy = np.flipud(image_numpy)
                    self.vis.image(image_numpy.transpose([2,0,1]), opts=dict(title=label),
                                       win=self.display_id + idx)
                    idx += 1

        if self.use_html: # save images to a html file
            for label, image_numpy in visuals.items():
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors_G(self, epoch, counter_ratio, errors):
        name = 'G'
        offset = 0
        if not hasattr(self, 'plot_data_G'):
            self.plot_data_G = {'X':[],'Y':[], 'legend':list(errors.keys())}
        self.plot_data_G['X'].append(epoch + counter_ratio)
        self.plot_data_G['Y'].append([errors[k] for k in self.plot_data_G['legend']])

        self.vis.line(
            X=np.stack([np.array(self.plot_data_G['X'])]*len(self.plot_data_G['legend']),1),
            Y=np.array(self.plot_data_G['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data_G['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id + offset)

    def plot_current_errors_D(self, epoch, counter_ratio, errors):
        name = 'D'
        offset = 20
        if not hasattr(self, 'plot_data_G'):
            self.plot_data_D = {'X':[],'Y':[], 'legend':list(errors.keys())}
        self.plot_data_D['X'].append(epoch + counter_ratio)
        self.plot_data_D['Y'].append([errors[k] for k in self.plot_data_D['legend']])

        self.vis.line(
            X=np.stack([np.array(self.plot_data_D['X'])]*len(self.plot_data_D['legend']),1),
            Y=np.array(self.plot_data_D['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data_D['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id + offset)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
    # def plot_current_label(self, labels, epoch):
    #     c_real, c_fake = labels
    #     c_real=c_real.data[0].view(self.option.ncond).cpu().numpy()+1
    #     c_fake=c_fake.data[0].view(self.option.ncond).cpu().numpy()+1
    #     labels=[c_real,c_fake]
    #     # print(labels)
    #     # self.vis.bar(
    #     #     X=c_real,
    #     #     opts=dict(
    #     #         title=self.name + 'real labels',
    #     #         stacked=False,
    #     #         legend=['real','fake'],
    #     #         rownames=['Male','Young','Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair','Straight_Hair','Wavy_Hair']
    #     #     ),
    #     #     win=self.display_id+10
    #     # )
    #     # print(labels)
    #     self.vis.bar(
    #         X=np.transpose(labels),
    #         opts=dict(
    #             title=self.name + ' real & fake labels',
    #             stacked=False,
    #             legend=['real','fake'],
    #             # rownames=['Male','Young','Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair','Straight_Hair','Wavy_Hair']
    #         ),
    #         win=self.display_id+10
    #     )




    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
