# -*- coding: utf-8 -*-
import sampler, pylab, os
import seaborn as sns
sns.set(font_scale=2)
sns.set_style("white")

def plot_kde(data, dir=None, filename="kde", color="Greens"):
	if dir is None:
		raise Exception()
	try:
		os.mkdir(dir)
	except:
		pass
	fig = pylab.gcf()
	fig.set_size_inches(16.0, 16.0)
	pylab.clf()
	bg_color  = sns.color_palette(color, n_colors=256)[0]
	ax = sns.kdeplot(data[:, 0], data[:,1], shade=True, cmap=color, n_levels=20, clip=[[-4, 4]]*2)
	ax.set_axis_bgcolor(bg_color)
	kde = ax.get_figure()
	kde.savefig("{}/{}.png".format(dir, filename))

def plot_scatter(data, dir=None, filename="scatter", color="blue"):
	if dir is None:
		raise Exception()
	try:
		os.mkdir(dir)
	except:
		pass
	fig = pylab.gcf()
	fig.set_size_inches(16.0, 16.0)
	pylab.clf()
	pylab.scatter(data[:, 0], data[:, 1], s=20, marker="o", edgecolors="none", color=color)
	pylab.savefig("{}/{}.png".format(dir, filename))

def main():
	scatter = sampler.gaussian_mixture(10000, num_cluster=8, scale=2, std=0.2)
	plot_scatter(scatter, "./plot")
	plot_kde(scatter, "./plot")

if __name__ == "__main__":
	main()
