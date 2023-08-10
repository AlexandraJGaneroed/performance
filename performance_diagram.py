import numpy as N
from matplotlib.axes import Axes
from matplotlib import pyplot as plt
from typing import Optional, List, Union
from fractions import Fraction
from matplotlib.lines import Line2D

class PerformanceDiagram(object):
    
    def __init__(
            self, ax: Optional[Axes] = None,
            bias_values: Optional[List[float]] = None,
            perf_values: Optional[List[float]] = None,
            line_step: Optional[float] = 0.01,
            bias_lines: Union[bool, dict] = True,
            bias_labels: Union[bool, dict] = True,
            perf_lines: Union[bool, dict] = True,
            perf_labels: Union[bool, dict] = True,
            **kwargs
            ):
        self.ax = ax if ax is not None else plt.gca()
        self.bias_values = N.array([0.25,0.5,1.0,2.0,4.0]) if bias_values is None else N.array(bias_values)
        self.perf_values = N.arange(0.1,1.,0.1) if perf_values is None else N.array(perf_values)
        self.line_step = 0.01 if line_step is None else line_step
        self.bias_legend = None
        self.perf_line_legend = []
        self.perf_fill_legend = []

        self.create_axis(**kwargs)

        self.plot_diagram(
            bias_lines=bias_lines,
            bias_labels=bias_labels,
            perf_lines=perf_lines,
            perf_labels=perf_labels,
        )

    def create_axis(self,
                    xlabel: str = 'Success Ratio (1-FAR)',
                    ylabel: str = 'Probability of Detection (POD)',
                    title: str = 'Performance Diagram',
                    ticks: Optional[List[float]] = None,
                    ticklabels: Optional[List[str]] = None,
                    aspect: Optional[str] = 'equal'
                    ):
        self.ax.grid(False)

        self.ax.set_xlim([0,1])
        ticks = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1.] if ticks is None else ticks
        ticklabels = ['{:1.1f}'.format(t) for t in ticks] if ticklabels is None else ticklabels
        self.ax.set_xticks(ticks)
        self.ax.set_xticklabels(ticklabels)
        self.ax.set_xlabel(xlabel)

        self.ax.set_ylim([0,1])
        self.ax.set_yticks(ticks)
        self.ax.set_yticklabels(ticklabels)
        self.ax.set_ylabel(ylabel)

        self.ax.set_title(title)
        if aspect is not None:
            self.ax.set_aspect(aspect)

    def plot_bias_lines(self, lw=1, color=[.2,.2,.2], linestyle='--', **kwargs):
        # plot bias lines as segments
        for b in self.bias_values:
            handle = self.ax.plot([0,1/b],[0,1], color=color, lw=lw, linestyle=linestyle, **kwargs)
            self.bias_legend = handle[0]

    def plot_bias_labels(self, percent=1., color=[.2,.2,.2], fontsize=8, bbox=dict(fc='white',color='white',pad=0), **kwargs):
        # plot bias labels at percent of the line length
        for b in self.bias_values:
            # compute the coordinates of the label location based on the percent of the line length
            if b>=1:
                x = 1/b*percent
                y = 1*percent
            else:
                x = 1*percent
                y = 1*b*percent

            # fromat the label string as integer ratio is less than 1
            bstr = str(Fraction(b)) if b<1 else '{:0.0f}x'.format(b)
            self.ax.annotate(bstr,xy=(x,y),
                                xycoords='data',color=color,
                                fontsize=fontsize,
                                bbox=bbox,
                                ha='center', va='center',
                                rotation=180.0*(N.arctan(b)/N.pi),
                                **kwargs
                                )

    def plot_perf_lines(self, perf='csi', fill_color=False, fill_alpha=0.5, fill_below=False, color=[.2,.2,.2], lw=1, **kwargs):
        """
        Plot perfomance lines (CSI/F1) as segments optionally filled with color map
        """
        # append a value of 1e-16 to the beginning of the array to avoid a divide by zero error
        # when computing the POD values if fill_below is True
        perf_values = N.insert(self.perf_values,0,1e-16) if fill_below else self.perf_values
        label = 'CSI' if perf == 'csi' else 'F1'
        podfunc = self.pod if perf == 'csi' else self.pod_f1
        
        for i,c in enumerate(perf_values):
            sr_x = N.arange(podfunc(c,0),1+self.line_step,self.line_step)
            pod_y = podfunc(c, 1-sr_x)
            handle = self.ax.plot(sr_x,pod_y,color=color, lw=lw, **kwargs)
            handle[0].set_label(label)
            if i == 0:
                self.perf_line_legend.append(handle[0])
            if fill_color:
                clr = fill_color if type(fill_color) is str else fill_color(c)
                alpha = c if type(fill_color) is str else fill_alpha
                handle = self.ax.fill_between(sr_x,pod_y,1,color=clr,alpha=alpha)
                handle.set_label(f'{label} ({c:1.1f})')
                self.perf_fill_legend.append(handle)

    def plot_perf_labels(self, perf='csi', percent=0.65, color=[.2,.2,.2], fontsize=8, bbox=dict(fc='white',color='white',pad=0), **kwargs):
        # plot perf labels at percent of the line length
        podfunc = self.pod if perf == 'csi' else self.pod_f1
        for c in self.perf_values:
            cstr = '{:1.1f}'.format(c)
            sr_x = N.arange(podfunc(c,0),1+self.line_step,self.line_step)
            pod_y = podfunc(c, 1-sr_x)
            x_ind = int(len(sr_x)*percent)
            y_ind = int(len(pod_y)*percent)
            slope = (pod_y[y_ind]-pod_y[y_ind-1])/(sr_x[x_ind]-sr_x[x_ind-1])
            x = sr_x[x_ind]
            y = pod_y[y_ind]
            self.ax.annotate(cstr,xy=(x,y),
                             xycoords='data',color=color,
                             fontsize=fontsize,
                             bbox=bbox,
                             ha='center', va='center',
                             rotation=180.0*(N.arctan(slope)/N.pi),
                             **kwargs
                            )

    def plot_diagram(self,
             bias_lines: Union[bool, dict] = True,
             bias_labels: Union[bool, dict] = True,
             perf_lines: Union[bool, dict] = True,
             perf_labels: Union[bool, dict] = True
             ):
        if bias_lines is True:
            self.plot_bias_lines()
        elif type(bias_lines) is dict:
            self.plot_bias_lines(**bias_lines)

        if bias_labels is True:
            self.plot_bias_labels()
        elif type(bias_labels) is dict:
            self.plot_bias_labels(**bias_labels)

        if perf_lines is True:
            self.plot_perf_lines()
        elif type(perf_lines) is dict:
            self.plot_perf_lines(**perf_lines)

        if perf_labels is True:
            self.plot_perf_labels()
        elif type(perf_labels) is dict:
            self.plot_perf_labels(**perf_labels)

    def plot_data(self, far=None, pod=None, csi=None, sr=None,
                  legend_elements: Optional[Union[List[dict],dict]] = None,
                  **legend_kw
                  ):
        """
        Plot data points on the performance diagram
        At least two between CSI, FAR/SR and POD must be provided (FAR and SR count as one)
        If FAR/SR and POD are provided, CSI will be checked for consistency.
        If FAR/SR is not provided, it will be computed from POD and CSI.
        If POD is not provided, it will be computed from FAR/SR and CSI.
        """
        # check for consistency of input data
        nr_provided = sum([far is not None or sr is not None, pod is not None, csi is not None])
        if nr_provided < 2:
            raise ValueError('At least two between CSI, FAR/SR and POD must be provided (FAR and SR count as one)')
        
        # if list type is provided, convert to numpy array
        if type(far) is list:
            far = N.array(far)
        if type(pod) is list:
            pod = N.array(pod)
        if type(csi) is list:
            csi = N.array(csi)
        if type(sr) is list:
            sr = N.array(sr)


        if far is None and sr is not None:
            far = 1 - sr
        
        if nr_provided == 2:
            if far is None:
                far = self.far(csi, pod)
            elif pod is None:
                pod = self.pod(csi, far)
            elif csi is None:
                csi = self.csi(far, pod)

        # plot the data points
        handles = []
        # check that far and pod are iterable
        if type(far) is float:
            far = [far]
        if type(pod) is float:
            pod = [pod]

        if legend_elements is None:
            for f,p in zip(far,pod):
                self.ax.plot(1-f,p,'o')
        else:
            if type(legend_elements) is dict:
                legend_elements = [legend_elements]
            for f,p,le in zip(far,pod,legend_elements):
                if 'marker' not in le:
                    handle = self.ax.plot(1-f,p,'o',**le)
                else:
                    handle = self.ax.plot(1-f,p,**le)
                handles.append(handle[0])

        # add legend elements
        full_legend = []
        if self.bias_legend is not None:
            self.bias_legend.set_label('Freq. BIAS')
            full_legend.append(self.bias_legend)

        if len(self.perf_fill_legend):
            full_legend += self.perf_fill_legend
        if len(self.perf_line_legend):
            full_legend += self.perf_line_legend

        full_legend += handles

        if len(full_legend):
            self.ax.legend(handles=full_legend, loc='upper left', bbox_to_anchor=(1.0, 1.0), **legend_kw)

    def far_pod_csi_sr_from_ct(self, TP, FP, FN):
        """
        Compute FAR, POD, CSI, and SR from contingency table (TP, FP, TN, FN)
        """
        FAR = FP/(TP+FP)
        POD = TP/(TP+FN)
        CSI = self.csi(FAR, POD)
        SR = 1 - FAR

        return FAR, POD, CSI, SR

    def csi(self, far, pod):
        """
        Compute CSI from FAR and POD

        REF: Gerapetritis, Harry, and Joseph M Pelissier.
        “THE CRITICAL SUCCESS INDEX AND WARNING STRATEGY.”
        In 17th Conference on Probablity and Statistics
            in the Atmospheric Sciences, Seattle, 2004.
        """
        return 1/( 1/(1-far) + 1/pod - 1 )

    def sr(self, csi, pod):
        """
        Compute SR from CSI and POD
        """
        return 1/( 1/csi - 1/pod + 1 )
    
    def far(self, csi, pod):
        """
        Compute FAR from CSI and POD
        """
        return 1 - self.sr(csi, pod)
    
    def pod(self, csi, far):
        """
        Compute POD (aka. recall or sensitivity) from CSI and FAR
        """
        return 1/( 1/csi - 1/(1-far) + 1)

    def f1(self, far, pod):
        """
        Compute F1 score from FAR and POD
        """
        return (2*(1-far)*pod) / (1-far+pod)

    def pod_f1(self, f1, far):
        """
        Compute POD (aka. recall or sensitivity) from F1 and FAR
        """
        return (f1*(1-far)) / (2*(1-far)-f1)