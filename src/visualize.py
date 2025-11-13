import matplotlib.pyplot as plt


PLOT_STYLE = {
    'figsize': (15, 5),
    'linewidth': 2,
    'markersize': 8,
    'grid_alpha': 0.3,
    'dpi': 300,
    'label_fontsize': 12,
    'title_fontsize': 14
}


def style_axis(ax, xlabel, ylabel, title, xticks=None, legend=False):
    ax.set_xlabel(xlabel, fontsize=PLOT_STYLE['label_fontsize'])
    ax.set_ylabel(ylabel, fontsize=PLOT_STYLE['label_fontsize'])
    ax.set_title(title, fontsize=PLOT_STYLE['title_fontsize'], fontweight='bold')
    ax.grid(True, alpha=PLOT_STYLE['grid_alpha'])
    if xticks:
        ax.set_xticks(xticks)
    if legend:
        ax.legend(fontsize=PLOT_STYLE['label_fontsize'])


def save_figure(filename, output_dir='results/plots'):
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename}', dpi=PLOT_STYLE['dpi'], bbox_inches='tight')
    plt.close()


def create_plot(config, output_dir='results/plots'):
    nrows = config.get('nrows', 1)
    ncols = config.get('ncols', len(config['subplots']))
    _, axes = plt.subplots(nrows, ncols, figsize=config.get('figsize', PLOT_STYLE['figsize']))

    if ncols == 1 and nrows == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for i, subplot in enumerate(config['subplots']):
        ax = axes[i] if ncols > 1 or nrows > 1 else axes[0]

        for series in subplot['series']:
            ax.plot(series['x'], series['y'],
                   marker=series.get('marker', 'o'),
                   linewidth=series.get('linewidth', PLOT_STYLE['linewidth']),
                   markersize=series.get('markersize', PLOT_STYLE['markersize']),
                   color=series.get('color'),
                   label=series.get('label'))

        style_axis(ax, subplot['xlabel'], subplot['ylabel'], subplot['title'],
                  xticks=subplot.get('xticks'), legend=subplot.get('legend', False))

    save_figure(config['filename'], output_dir)
