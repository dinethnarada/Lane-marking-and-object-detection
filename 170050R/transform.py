import math

# Function - Function to resize the image
# Params - Input img, Scale factor
# Return - Scaled Image
def resize(original_img, scale=0.8):
	oh, ow = len(original_img), len(original_img[0])
	nh, nw = int(oh*scale), int(ow*scale)
	output_img = [[None]*nw for _ in range(nh)]

	# Fill out all the output_img values with mapped real values
	for i in range(nh):
		for j in range(nw):
			# map the coordinates back to the original image
			x = i / scale
			y = j / scale
			# Calculate the 4 neigbourhood pixel values
			x_f = math.floor(x)
			x_c = min(oh - 1, math.ceil(x))
			y_f = math.floor(y)
			y_c = min(ow - 1, math.ceil(y))
			# Use linear thresolding method for measure correct value
			if (x_c == x_f) and (y_c == y_f):
				value = original_img[int(x)][int(y)]
			elif (y_c == y_f):
				q1 = original_img[int(x_f)][int(y)]
				q2 = original_img[int(x_c)][int(y)]
				value = (q1 * (x_c - x)) + (q2	 * (x - x_f))
			elif (x_c == x_f):
				q1 = original_img[int(x)][int(y_f)]
				q2 = original_img[int(x)][int(y_c)]
				value = q1 * (y_c - y) + q2 * (y - y_f)
			else:
				q1 = original_img[x_f][y_f] * (x_c - x) + original_img[x_c][y_f] * (x - x_f)
				q2 = original_img[x_f][y_c] * (x_c - x) + original_img[x_c][y_c] * (x - x_f)
				value = q1 * (y_c - y) + q2 * (y - y_f)
			output_img[i][j] = value
	return output_img