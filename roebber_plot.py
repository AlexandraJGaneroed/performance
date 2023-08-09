"""
Roebber plot.   Code based on https://johnrobertlawson.github.io/evac/_modules/evac/plot/performance.html
"""
import numpy as N
from matplotlib.axes import Axes
from matplotlib import pyplot as plt
from typing import Optional, List, Union

class PerformanceDiagram(object):
    
    def __init__(
            self, ax: Optional[Axes] = None,
            bias_values: Optional[List[float]] = None,
            csi_values: Optional[List[float]] = None,
            line_step: Optional[float] = 0.01,
            bias_lines: Union[bool, dict] = True,
            bias_labels: Union[bool, dict] = True,
            csi_lines: Union[bool, dict] = True,
            csi_labels: Union[bool, dict] = True,
            **kwargs
            ):
        self.ax = ax if ax is not None else plt.gca()
        self.bias_values = N.array([0.25,0.5,1.0,2.0,4.0]) if bias_values is None else N.array(bias_values)
        self.csi_values = N.arange(0.1,1.,0.1) if csi_values is None else N.array(csi_values)
        self.line_step = 0.01 if line_step is None else line_step

        self.create_axis(**kwargs)

        self.plot(
            bias_lines=bias_lines,
            bias_labels=bias_labels,
            csi_lines=csi_lines,
            csi_labels=csi_labels,
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
            self.ax.plot([0,1/b],[0,1], color=color, lw=lw, linestyle=linestyle, **kwargs)

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

            self.ax.annotate('{:1.1f}'.format(b),xy=(x,y),
                                xycoords='data',color=color,
                                fontsize=fontsize,
                                bbox=bbox,
                                ha='center', va='center',
                                rotation=180.0*(N.arctan(b)/N.pi),
                                **kwargs
                                )

    def plot_csi_lines(self, fill_color=False, fill_alpha=0.5, fill_below=False, color=[.2,.2,.2], lw=1, **kwargs):
        """
        Plot CSI lines as segments optionally filled with color map
        """
        # append a value of 1e-16 to the beginning of the array to avoid a divide by zero error
        # when computing the POD values if fill_below is True
        csi_values = N.insert(self.csi_values,0,1e-16) if fill_below else self.csi_values
        for c in csi_values:
            sr_x = N.arange(c,1+self.line_step,self.line_step)
            pod_y = self.pod(c, 1-sr_x)
            self.ax.plot(sr_x,pod_y,color=color, lw=lw, **kwargs)
            if fill_color:
                clr = fill_color if type(fill_color) is str else fill_color(c)
                alpha = c if type(fill_color) is str else fill_alpha
                self.ax.fill_between(sr_x,pod_y,1,color=clr,alpha=alpha)

    def plot_csi_labels(self, percent=0.65, color=[.2,.2,.2], fontsize=8, bbox=dict(fc='white',color='white',pad=0), **kwargs):
        # plot csi labels at percent of the line length
        for c in self.csi_values:
            cstr = '{:1.1f}'.format(c)
            sr_x = N.arange(c,1+self.line_step,self.line_step)
            pod_y = self.pod(c, 1-sr_x)
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
    
    def plot(self,
             bias_lines: Union[bool, dict] = True,
             bias_labels: Union[bool, dict] = True,
             csi_lines: Union[bool, dict] = True,
             csi_labels: Union[bool, dict] = True
             ):
        if bias_lines is True:
            self.plot_bias_lines()
        elif type(bias_lines) is dict:
            self.plot_bias_lines(**bias_lines)

        if bias_labels is True:
            self.plot_bias_labels()
        elif type(bias_labels) is dict:
            self.plot_bias_labels(**bias_labels)

        if csi_lines is True:
            self.plot_csi_lines()
        elif type(csi_lines) is dict:
            self.plot_csi_lines(**csi_lines)

        if csi_labels is True:
            self.plot_csi_labels()
        elif type(csi_labels) is dict:
            self.plot_csi_labels(**csi_labels)

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
        Compute POD from CSI and FAR
        """
        return 1/( 1/csi - 1/(1-far) + 1)
