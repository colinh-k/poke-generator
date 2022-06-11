import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def plot_batch(imgs, title, fname, n=64):
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.title(title)
    plt.imshow(
        make_grid(imgs[:n], padding=2, normalize=True).numpy().transpose((1,2,0))
    )

    print(f'saving image to {fname}')
    plt.savefig(fname)

# plots x on x axis, and y on y axis
def plot_line(x, y, fname, label=None, xaxis=None, yaxis=None, title=None, fign=None, legend=True, ltype='-'):
    plt.figure(fign)
    plt.plot(x, y, ltype, label=label)

    if legend:
        plt.legend()
    if xaxis:
        plt.xlabel(xaxis)
    if yaxis:
        plt.ylabel(yaxis)
    if title:
        plt.title(title)

    plt.savefig(fname)
    # else:
    #     plt.show()

def plot_results(x, y1, y1_label, y2, y2_label, xaxis, yaxis, title, fname):
    plot_line(x, y1, fname, y1_label, xaxis, yaxis, title, fign=0)
    plot_line(x, y2, fname, y2_label, fign=0)