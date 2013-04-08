
from optparse import OptionParser
import re
import os

import numpy as np
import pylab as plt

from gwt_ka.data_process import GWTReferrals

# you can modify this regular expression to define your own branded keywords
re_branded = re.compile('seomoz|page authority')

# branded_keyword is a callable that takes a search query and
# checks whether it is branded or not
is_branded_keyword = lambda x: bool(re_branded.search(x))

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-c", "--csv",
       dest="gwt_csv_file",
       help="CSV file from Google Webmaster Tools", metavar="FILE")
    parser.add_option("-o", "--output_directory",
        dest="outdir",
        help="Path to output directory")
    (options, args) = parser.parse_args()

    gwt = GWTReferrals.load_data(options.gwt_csv_file)
    outdir = options.outdir

    # plot univariates, save to files
    figs = gwt.plot_univariates()
    for f, k in zip(figs, [1, 2, 3, 4]):
        f.savefig(os.path.join(outdir, 'univariates_%s.png' % k))

    # plot an overall CTR curve
    ctr_position, ctr_curve = gwt.ctr_curve()
    fig = plt.figure(5)
    fig.clf()
    plt.plot(ctr_position[:20], ctr_curve[:20])
    plt.title("Click through rate")
    plt.ylabel("CTR")
    plt.xlabel("Average Search Position")
    fig.show()
    fig.savefig(os.path.join(outdir, 'ctr_position.png'))

    # get branded words
    all_queries = gwt.get_queries()
    branded_mask = np.array([is_branded_keyword(query) for query in all_queries])
    branded_queries = [all_queries[k]
                for k in xrange(len(all_queries)) if branded_mask[k]]

    # write branded queries to a file for inspection
    with open(os.path.join(outdir, 'all_branded_queries.txt'), 'w') as fout:
        fout.write('\n'.join(
                [all_queries[k] for k in xrange(len(all_queries))
                    if branded_mask[k]]))

    # make univariate plots of branded vs not branded
    figs = gwt.plot_univariates([branded_mask, ~branded_mask], first_fig_id=6,
                labels=['Branded','Not Branded'])
    for f, k in zip(figs, [1, 2, 3, 4]):
        f.savefig(os.path.join(outdir, 'univariates_branded_not_branded_%s.png' % k))


