"""Classification reports utilities."""

from warnings import warn
from os.path import exists
import json
from imblearn.metrics import classification_report_imbalanced
import pandas as pd
import numpy as np
from .display import md

class ClfReport:
    """Stores and display batches of classification metrics.

    Parameters
    ----------
    persist : string, optional
        Path to the file where to store the reports (json). If the file  already
        exists, saved reports are loaded from it unless *reset* is True. If set,
        reports are saved each time a report is computed.
    reset : bool, optional
        Clear saved reports.

    Examples
    --------
    Records and display 3 reports, assuming (Xs,Xt,ys,yt) is train test data.

    >>> from ozpydry.metrics import ClfReport
    >>> report = ClfReport('kNC-reports')
    >>> for i in [5, 10, 15]
    >>>     knc = KNeighborsClassifier(i)
    >>>     knc.fit(Xs)
    >>>     report(knc, (Xs,Xt,ys,yt), 'Knc(%d)' % i, False)
    >>> report.show()
    """

    def __init__(self, persist=None, reset=False):
        self.persist_ = persist
        self.reports_ = dict()
        if persist is not None and not reset:
            self.load(persist)

    def __call__(self, estimator, tts, label=None, show=True):
        """Computes and displays classification metrics.

        Parameters
        ----------
        estimator : estimator object
            This is assumed to implement the scikit-learn estimator interface and
            should be prefited.
        tts : tuple|list of array-like
            Typically it should match the return of `train_test_split` function,
            ie. (Xs, Xt, ys, yt)
        label : string, optional
            Identifier of the estimator's report, default to `estimator.__repr__`.
            .. note:: This overrite a report with the same label.
        show : bool, optional
            Whether to immediately display the report.

        Returns
        -------
        self : ClfReport
            If `show` param is True nothing is return to prevent printing
            object's pointer.
        """
        Xs, Xt, ys, yt = tts
        id = label or str(estimator)
        preds = estimator.predict(Xt)
        fnrep = classification_report_imbalanced
        stats = fnrep(preds, yt, output_dict=True)
        cols = ['pre', 'rec', 'spe', 'f1', 'geo', 'iba', 'sup']
        tact = ys.value_counts(normalize=True).round(4) * 100
        self.reports_[id] = {
            'daset': [*Xs.shape, *tact.values],
            'stats': [
                ['0', *[np.round(stats[0][k], 2) for k in cols]],
                ['1', *[np.round(stats[1][k], 2) for k in cols]],
                ['avg / total', *[np.round(stats['avg_' + k], 2) for k in cols[:-1]], ''],
                ['test accuracy','', '', '', np.round(estimator.score(Xt, yt), 2), '', '', ''],
                ['train accuracy','', '', '', np.round(estimator.score(Xs, ys), 2), '', '', '']]}
        # Save if needed
        if self.persist_:
            self.save()
        if show:
            self.show(id)
        else: # prevent pointer printing
            return self

    def tone_(self, v):
        """Background style for colored cell"""
        return f'background:hsl(calc((({v} - .5) / .5) * 120), 70%, 85%, calc({v}));'

    def tomdc_(self, reps, tone=False, header=None):
        """Markdown compact display"""
        mkt = self.tone_ if tone else (lambda v: "")
        out = "|%s|test<br>acc.|train<br>acc.|0<br>pre&nbsp; . &nbsp;rec&nbsp; . &nbsp;&nbsp;f1|1<br>pre&nbsp; . &nbsp;rec&nbsp; . &nbsp;&nbsp;f1|<br>%% bal.|<br>obs|\n" % (header.replace('\n',"<br>") if header is not None else "")
        out+= "|-|:-:|:-:|:-:|:-:|:-:|:-|\n"
        mks = lambda v: f'background:hsl(calc((({v} - .5) / .5) * 120), 70%, 85%, calc({v}));'
        for k, r in reps:
            s, f, a, b = r['daset'] # dataset balancing stats
            cl = 'c' if tone else ""
            vals = [[v if isinstance(v, str) else \
                '<span class="v %s" style="%s">%.2f</span>' % (cl,mkt(v),v) for v in ss] for ss in r['stats']]
            out+= "|%s|%s|%s|%s&nbsp;&nbsp;%s&nbsp;&nbsp;%s|%s&nbsp;&nbsp;%s&nbsp;&nbsp;%s|<small>%.f / %.f</small>|<small>%d</small>|\n" % (k,
                vals[3][4], vals[4][4],
                vals[0][1], vals[0][2], vals[0][4],
                vals[1][1], vals[1][2], vals[1][4],
                a, b, s)
        out = f'<div class="rt">\n\n{out}\n</div>'
        out+= """
        <style>
            .rt { color:#444; }
            .rt .v { padding:2px; font-style:normal; }
            .rt th, .rt td { border:none; border-bottom: 1px solid lightgrey; }
            .rt th, .rt td { border-right: 4px solid white; }
            .rt tr:first-child th:first-child { font-size:.8em;font-weight:normal;text-align:left; }
            .rt td:first-child { text-align:right; padding:0 10px; }
            .rt tr:first-child th + th { font-weight:normal;color:black;border-bottom-color:#888; }
        </style>
        """
        return out

    def tomdf_(self, reps=None, tone=False, header=None):
        """Markdown full display"""
        mkt = self.tone_ if tone else (lambda v: "")
        out = "|%s||pre|rec|spe|f1|geo|iba|sup|\n" % (header.replace('\n',"<br>") if header is not None else "")
        out+= "|-|-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|\n"
        for k, r in reps:
            s, f, a, b = r['daset'] # dataset balancing stats
            out+= "|%s|||||||||\n" % k
            cl = 'c' if tone else ""
            vals = [['<i class="v %s" style="%s">%.2f</i>' % (cl,mkt(v),v) if isinstance(v, float) else \
                        str(v) for v in ss] for ss in r['stats']]
            for i,l in enumerate(vals):
                pre = ""
                if i == 0:
                    pre = "_%.f%% balance_" % a
                if i == 1:
                    pre = "_%.f%% balance_" % b
                # out+= "|" + pre + "|" + "|".join(map(lambda x: format(x, '.2f') if isinstance(x,float) else str(x), l)) + "|\n"
                out+= "|" + pre + "|" + "|".join(l) + "|\n"
        out = f'<div class="rt">\n\n{out}\n</div>'
        out+= """
        <style>
            .rt { color:#444; }
            .rt .v { padding:2px; font-style:normal; }
            .rt th, .rt td { border:none; border-bottom: 1px solid lightgrey; }
            .rt tr:first-child th:first-child { font-size:.8em;font-weight:normal;text-align:left; }
            .rt tr:nth-child(6n+1) {color:black;}
            .rt th + th, .rt td + td { border-right: 4px solid white; }
            .rt tr:first-child th + th + th { font-weight:normal;color:black;border-bottom-color:#888; }
        </style>
        """
        return out

    def preps_(self, include=None):
        """Select reports"""
        # filter reports to display
        if isinstance(include, str):
            include = [include]

        return [(k, self.reports_[k]) for k in include] \
                    if hasattr(include, "__iter__") else self.reports_.items()


    def show(self, include=None, compact=False, title=None, tone=False):
        """Display classification reports.

        Parameters
        ----------
        include : string or string-list, optional
            If set, display specified report(s) otherwise all
        compact : bool, optional
            Display in compact mode, one line by report
        tone : bool, optional
            Highlights values with colors
        """

        reps = self.preps_(include=include)
        md(self.tomdc_(reps, tone, title) if compact else self.tomdf_(reps, tone, title))

    def load(self, path):
        """Load reports from a file.

        The loaded reports will replace current if any. Note that loading reports
        won't flag this instance as persistent. See `__init__` and `save`.
        .. note:
            the method silently fail if the file did not exists.

        Parameters
        ----------
        path : string
            Path of the json encoded file to load.
        """
        if exists(path):
            self.reports_ = json.load(open(path))

    def save(self, path=None):
        """Saves computed reports.

        Parameters
        ----------
        path : string, optional
            File path to save the reports into. This ovewrite the given path
            at init time when instance is flaged as persistant see `__init__`
            .. note:: if the path contains folders, those must exists

        Warns
        -----
            Raise warning if no suitable path found.
        """
        if path is None:
            if  self.persist_ is not None:
                path = self.persist_
            else:
                warn("No path given nor at init time, reports cannot be saved.")

        def np_encoder(object):
            if isinstance(object, np.generic):
                return object.item()

        json.dump(self.reports_, open(path, "w"), default=np_encoder)

    def to_pandas(self):
        """Constructs a DataFrame from the computed reports.

        Returns
        -------
        pandas.DataFrame
        """
        col = ['Estimator','*','pre', 'rec', 'spe', 'f1', 'geo', 'iba', 'sup']
        mdf = pd.DataFrame(columns=col).reset_index(drop=True)
        for k, r in self.reports_.items():
            mdf.reset_index()
            rdf = pd.DataFrame([[k, *vs] for vs in r.get('stats')], columns=col)
            mdf = pd.concat([mdf, rdf])
            mdf[col[2:]] = mdf[col[2:]].apply(pd.to_numeric)

        return mdf.set_index(keys=['Estimator','*'])

    def to_markdown(self, include=None, compact=False, title=None, tone=False):
        """Returns reports as mardown.

        See `show` method for parameters
        """
        reps = self.preps_(include=include)
        return self.tomdc_(reps, tone, title) if compact else self.tomdf_(reps, tone, title)

    def _tostr(self, include=None):
        cols = ['pre', 'rec', 'spe', 'f1', 'geo', 'iba', 'sup']
        out = "{:<16}{:>16}{:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:^8}\n".format('','',*cols)
        tre = [(include, self.reports_[include])] if include else self.reports_.items()
        for k, r in tre:
            s, f, a, b = r['daset'] # samples, features, cl0, cl1
            out+= "\033[7m{:<30}\033[0m\n".format(k)
            for i, l in enumerate(r['stats']):
                pre = ""
                if i == 0:
                    pre = "%.f%% balance" % a
                if i == 1:
                    pre = "%.f%% balance" % b
                vals = map(lambda x: format(x, '.2f') if isinstance(x,float) else str(x), l)
                out+= "\033[3m{:<16}\033[0m{:>16} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:^8}\n".format(pre, *vals)
        return out

    def __str__(self):
        """Raw text representation of the reports."""
        return self._tostr()


def clf_report(estimator, tts, label=None):
    """Display classification report.

    Unlike ClfReport, this function does not store the computed report.
    See `ClfReport.__call__` for parameters description.

    Examples
    --------
    >>> from ozpydry.metrics import clf_report
    >>> lreg = LogisticRegression()
    >>> lreg.fit(Xs)
    >>> clf_report(lreg, (Xs,Xt,ys,yt), 'LogReg')
    """
    return ClfReport()(estimator, tts, label)