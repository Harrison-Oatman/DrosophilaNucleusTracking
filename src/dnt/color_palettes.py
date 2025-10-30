import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from types import SimpleNamespace

nc_pal = {10: "#1d4e89", 11: "#00b2ca", 12: "#7dcfb6", 13: "#fbd1a2", 14: "#f79256"}
trk_pal = ["steelblue", "gold"]
four_color_pal = ["#44af69", "#f8333c", "#fcab10", "#2b9eb3"]

def set_plot_style(update_rc = None):

    if update_rc is None:
        update_rc = {}

    rc = {
        "figure.facecolor": "#FFFFFF00",
        "axes.facecolor": "#FFFFFF00",
        "legend.framealpha": 0.2,
        "lines.color": "k",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "font.size": 10,
    }

    rc.update(update_rc)

    sns.set_context("paper")
    sns.set_theme(style="ticks", font="Arial", rc=rc)

palettes = SimpleNamespace(
    nc = nc_pal,
    trk = trk_pal,
    four_color = four_color_pal,
)


def make_colormap_figure(cmap, vmin, vmax, label):
    fig, ax = plt.subplots(figsize=(1.5, 4), dpi=100)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)  # Define the data range for the colorbar

    # make figure transparent
    fig.patch.set_alpha(0.0)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=ax, orientation='vertical', label=label)

    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('', fontsize=16, loc="center")

    # Adjust layout to prevent labels from overlapping
    plt.tight_layout()
    ax.invert_yaxis()

    return fig, ax