import math

def bl_resize(original_img, scale=0.8):
	oh, ow = len(original_img), len(original_img[0])
	nh, nw = int(oh*scale), int(ow*scale)
	output_img = [[None]*nw for _ in range(nh)]

	for i in range(nh):
		for j in range(nw):
			#map the coordinates back to the original image
			x = i / scale
			y = j / scale
			#calculate the coordinate values for 4 surrounding pixels.
			x_f = math.floor(x)
			x_c = min(oh - 1, math.ceil(x))
			y_f = math.floor(y)
			y_c = min(ow - 1, math.ceil(y))

			if (x_c == x_f) and (y_c == y_f):
				value = original_img[int(x)][int(y)]
			elif (x_c == x_f):
				q1 = original_img[int(x)][int(y_f)]
				q2 = original_img[int(x)][int(y_c)]
				value = q1 * (y_c - y) + q2 * (y - y_f)
			elif (y_c == y_f):
				q1 = original_img[int(x_f)][int(y)]
				q2 = original_img[int(x_c)][int(y)]
				value = (q1 * (x_c - x)) + (q2	 * (x - x_f))
			else:
				v1 = original_img[x_f][y_f]
				v2 = original_img[x_c][y_f]
				v3 = original_img[x_f][y_c]
				v4 = original_img[x_c][y_c]

				q1 = v1 * (x_c - x) + v2 * (x - x_f)
				q2 = v3 * (x_c - x) + v4 * (x - x_f)
				value = q1 * (y_c - y) + q2 * (y - y_f)

			output_img[i][j] = value

	print(max(max(x) for x in output_img))
	print(min(min(x) for x in output_img))
	return output_img