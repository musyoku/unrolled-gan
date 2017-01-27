import pylab
import numpy as np
from StringIO import StringIO
from PIL import Image

def tile_binary_images(x, dir=None, filename="x", row=10, col=10):
	if dir is None:
		raise Exception()
	try:
		os.mkdir(dir)
	except:
		pass
	fig = pylab.gcf()
	fig.set_size_inches(col * 2, row * 2)
	pylab.clf()
	pylab.gray()
	for m in range(row * col):
		pylab.subplot(row, col, m + 1)
		pylab.imshow(np.clip(x[m], 0, 1), interpolation="none")
		pylab.axis("off")
	pylab.savefig("{}/{}.png".format(dir, filename))

def tile_rgb_images(x, dir=None, filename="x", row=10, col=10):
	if dir is None:
		raise Exception()
	try:
		os.mkdir(dir)
	except:
		pass
	fig = pylab.gcf()
	fig.set_size_inches(col * 2, row * 2)
	pylab.clf()
	for m in range(row * col):
		pylab.subplot(row, col, m + 1)
		pylab.imshow(np.clip(x[m], 0, 1), interpolation="none")
		pylab.axis("off")
	pylab.savefig("{}/{}.png".format(dir, filename))