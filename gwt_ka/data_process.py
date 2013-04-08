
import re
from itertools import izip

import numpy as np
import pylab as plt

class GWTReferrals(object):

    def __init__(self, data):
        """Don't use constructor directly, use load_data instead

        data = [[query1, impression1, click1, ctr1, avg position1],
                [query2, ...]]

            use -1 to signify missing data
        """
        self._queries = [ele[0] for ele in data]
        self._data = np.array([tuple(ele[1:]) for ele in data],
                        dtype=[('impressions', np.float),
                               ('clicks', np.float),
                               ('ctr', np.float),
                               ('avg_position', np.float)])

        # fill missing data
        # (1) compute a CTR curve based on average position and use it to
        #     filling missing CTR
        mask = self._data['ctr'] >= 0
        if not mask.all():

            ctr_position, ctr_curve = self.ctr_curve(mask)

            # sometimes the CTR curve can have nan in it
            # in cases where we only observe one long tail keywords
            # for a certain position.  interpolate them in that case

            ctr_missing = np.isnan(ctr_curve)
            ctr_curve[ctr_missing] = np.interp(
                        ctr_position[ctr_missing],
                        ctr_position[~ctr_missing],
                        ctr_curve[~ctr_missing])

            missing_position_index = self.get_position()[~mask] - 1
            self._data['ctr'][~mask] = ctr_curve[missing_position_index]

            # (2) fill missing # clicks with the impressions and CTR
            self._data['clicks'][~mask] = self._data['impressions'][~mask] \
                                            * self._data['ctr'][~mask]

        # some meta data
        self._meta = {'avg_position':'Avg. Position, weighted by clicks',
                      'clicks':'Clicks',
                      'ctr':'CTR, weighted by impressions',
                      'impressions':'Impressions'}
        self._ranges = {'avg_position':[0, 20], 
                        'ctr':[0, 1],
                        'clicks':[0, 50],
                        'impressions':[0, 500]}

    @classmethod
    def load_data(cls, gwt_csv_file, less_then_ten_replace=3):
        """Load the data from a file.

        less_then_ten_replace = use this value for number of clicks
            in the case where GWT returns "<10"
        """
        import csv

        strip_comma_cast_int = lambda x: int(re.sub(",", "", x))

        data = []
        with open(gwt_csv_file, 'rb') as f:
            f.readline()   # skip the header
            csv_reader = csv.reader(f, delimiter=',', quotechar='"')
            for query, impress, click, c, pos in csv_reader:

                # parse the data into numbers
                # replace "<10"
                if impress.strip() == "<10":
                    # here we set ctr to -1 as missing, then replace it later
                    # with with an average
                    impress = less_then_ten_replace
                    click = -1
                    c = -1
                elif click.strip() == "<10":
                    # have some impressions, need to remove commas
                    impress = strip_comma_cast_int(impress)
                    click = less_then_ten_replace
                    c = float(click) / impress
                else:
                    # have data for all impression, clicks, ctr
                    impress = strip_comma_cast_int(impress)
                    click = strip_comma_cast_int(click)
                    c = float(c.strip().rstrip("%")) / 100.0

                # avg position
                pos = float(pos)

                data.append([query, impress, click, c, pos])

        return cls(data)

    def get_position(self):
        return np.round(self._data['avg_position']).astype(np.int)

    def get_queries(self):
        return self._queries

    def ctr_curve(self, mask=None):
        """Compute a CTR curve as a function of search position

        mask = a length(data) boolean list whether to include
            the data point in the calculation or not.
        
        returns position, ctr
            where position is 1..N and CTR is a length N vector with the
            CTR at the corresponding position.

            If there is no data at the position, return np.nan in the CTR curve

        Weights the CTR curve by number impressions
        """
        position_index = self.get_position() - 1
        counts = np.zeros(position_index.max() + 1)
        ctr = np.zeros(position_index.max() + 1)
        if mask is not None:
            # we have a mask
            # CTR = total clicks / total impressions
            for p, c, i, m in izip(position_index,
                         self._data['clicks'],
                         self._data['impressions'],
                         mask):
                if m:
                    counts[p] += i
                    ctr[p] += c
        else:
            # no mask
            for p, c, i in izip(position_index, self._data['clicks'], self._data['impressions']):
                counts[p] += i
                ctr[p] += c

        # compute average CTR
        valid_data = counts > 0
        ctr[valid_data] = ctr[valid_data] / counts[valid_data]
        ctr[~valid_data] = np.nan

        return np.arange(position_index.max() + 1) + 1, ctr

    def plot_univariates(self, mask=None, labels=None,
            nbins=30, ranges=None, first_fig_id=1):
        """Make some plots of the univariates

        mask = a list of masks for different categories to overlay
            or None to plot everything
        labels = list of strings to label the plots
        nbins = use this many bins for the histogram
        ranges = a dictionary specifying the ranges for the histograms,
            e.g. {'avg_position':[0, 25]}

        Returns a list of fig objects"""
        # this code is a bit ugly...
        ret = []
        k = first_fig_id

        if mask is None:
            sp = 220
            mask = [np.array([True] * len(self._data))]
            fig = plt.figure(k)
            fig.clf()
        else:
            sp = len(mask) * 100 + 10


        for v, title in self._meta.iteritems():
            if len(mask) > 1:
                fig = plt.figure(k)
                fig.clf()
            else:
                plt.subplot(sp + k)

            kwargs = {'bins':nbins, 'normed':True}

            i = 1
            for m in mask:
                if len(mask) > 1:
                    plt.subplot(len(mask), 1, i)
                i += 1

                if v  == 'ctr':
                    kwargs['weights'] = self._data['impressions'][m]
                    avg = np.sum(self._data['impressions'][m] * self._data[v][m]) \
                                    / np.sum(self._data['impressions'][m])
                elif v == 'avg_position':
                    kwargs['weights'] = self._data['clicks'][m]
                    avg = np.sum(self._data['clicks'][m] * self._data[v][m]) \
                                    / np.sum(self._data['clicks'][m])
                else:
                    kwargs['weights'] = None
                    avg = np.sum(self._data[v][m]) / len(self._data[v][m])

                if ranges and v in ranges:
                    kwargs['range'] = ranges[v]
                elif v in self._ranges:
                    kwargs['range'] = self._ranges[v]
                else:
                    kwargs['range'] = None

                plt.hist(self._data[v][m], **kwargs)
                if len(mask) > 1:
                    if labels:
                        le = labels[i-2] + ' '
                    else:
                        le = ''
                    le += "$\mu$=%0.2f" % avg
                    plt.legend([le])
                    
            # the title
            if len(mask) == 1:
                ti = "%s; $\mu$=%0.2f" % (title, avg)
                plt.title(ti, fontsize=12)
            else:
                ti = title
                plt.figtext(0.5, 0.92, ti, ha='center', color='black', weight='bold')

            fig.show()
            if len(mask) > 1:
                ret.append(fig)

            k += 1

        if len(mask) == 1:
            ret.append(fig)

        return ret

