import numpy as np
from shapely.geometry import Point, Polygon
from shapely import prepared
from scipy.ndimage.filters import gaussian_filter


class ParticleDistribution:
    def __init__(self, img_size, density=0.5, size_2_density=0.5,
                 g_noise_mu=0.2, g_noise_sigma=0.1, min_intensity=0.0):
        self.img_size = img_size
        self.num_particles = 0
        self.img = []

        self.generate(density, size_2_density)
        self.add_gaussian_noise(g_noise_mu, g_noise_sigma)
        # self.gaussian_blur()
        self.normalize_image()
        self.brightness(min_intensity)

    def generate(self, density=0.5, size_2_density=0.5):

        img_size = self.img_size
        num_particles = int(img_size*img_size * density)

        # particle coordinates
        x = (np.random.rand(num_particles, 1) * (img_size-1)).astype(int)
        y = (np.random.rand(num_particles, 1) * (img_size-1)).astype(int)

        # probability distributions for size 2 particles
        prob_a = np.random.rand(num_particles)
        prob_b = np.random.rand(num_particles)

        # place size 1 and 2 particles in image
        p = np.zeros((img_size, img_size))
        for i in range(num_particles):
            # particles of size 1
            p[x[i], y[i]] = 1

            # particles of size 2
            if prob_a[i] < size_2_density:
                if prob_b[i] > 0.5:
                    if x[i] != (img_size-1):
                        p[x[i]+1, y[i]] = 1
                    else:
                        p[x[i]-1, y[i]] = 1
                else:
                    if y[i] != (img_size-1):
                        p[x[i], y[i]+1] = 1
                    else:
                        p[x[i], y[i]-1] = 1

        self.img = p

    def add_gaussian_noise(self, mu, sigma):
        #mu = 0.2 # 0.1
        #sigma = 0.1 # 0.05
        self.img = self.img + mu + \
                   sigma * np.random.randn(self.img.shape[0], self.img.shape[1])

    def gaussian_blur(self):
        self.img = gaussian_filter(self.img, sigma=0.5)

    def brightness(self, min_intensity):
        self.img = (1-min_intensity) * self.img + min_intensity

    def normalize_image(self):
        min_val, max_val = np.min(self.img), np.max(self.img)
        self.img = (self.img - min_val) / (max_val - min_val)

    def get_particle_image(self):
        return self.img


class Shape:

    def __init__(self):
        self.x = []
        self.y = []
        self.polygon = []
        self.pixel_coords = []

    def transform(self, scale, rot, x, y):
        x_rs = np.cos(rot) * np.multiply(self.x, scale) \
               - np.sin(rot) * np.multiply(self.y, scale)
        y_rs = np.sin(rot) * np.multiply(self.x, scale) \
               + np.cos(rot) * np.multiply(self.y, scale)
        self.x = x_rs + x
        self.y = y_rs + y

    def shapely_polygon(self):
        shapely_pnts = []

        for i in range(len(self.x)):
            shapely_pnts.append(Point(self.x[i], self.y[i]))

        self.polygon = Polygon(shapely_pnts)

    def pixels_within_polygon(self):

        prep = prepared.prep(self.polygon)

        x_min, y_min, x_max, y_max = self.polygon.bounds
        x_min, y_min, x_max, y_max = \
            int(x_min), int(y_min), int(x_max), int(y_max)
        bb_box = [[x, y] for x in range(x_min, x_max+1)
                  for y in range(y_min, y_max+1)]
        pixels = []
        for pixel_coord in bb_box:
            pt = Point(pixel_coord[0], pixel_coord[1])
            if prep.contains(pt):
                pixels.append([int(pixel_coord[0]), int(pixel_coord[1])])

        self.pixel_coords = pixels

    def get_pixel_form(self, scale, rot, x, y):
        self.transform(scale, rot, x, y)
        self.shapely_polygon()
        self.pixels_within_polygon()

    def put_shape_into_image(self, intensity, img, gradient_enable=False):

        gradient_center = np.random.randint(0, len(self.pixel_coords))
        x_g = self.pixel_coords[gradient_center][0]
        y_g = self.pixel_coords[gradient_center][1]

        if gradient_enable:
            dist = []
            for i in range(len(self.pixel_coords)):
                x, y = self.pixel_coords[i][0], self.pixel_coords[i][1]
                dist.append(np.sqrt((x_g-x)**2+(y_g-y)**2))

            max_len = np.max(dist)

        for i in range(len(self.pixel_coords)):
            x, y = self.pixel_coords[i][0], self.pixel_coords[i][1]
            if 0 <= x < img.shape[0] and 0 <= y < img.shape[1]:
                if gradient_enable:
                    img[x, y] = intensity * dist[i]/max_len
                else:
                    img[x, y] = intensity


class Triangle(Shape):

    def __init__(self):
        super().__init__()
        self.x = [-0.5, 0.5, 0, -0.5]
        self.y = [-0.5, -0.5, 0.5, -0.5]


class Rectangle(Shape):

    def __init__(self):
        super().__init__()
        self.x = np.multiply(0.5, [-1, 1, 1, -1, -1])
        self.y = np.multiply(0.25, [-1, -1, 1, 1, -1])


class Disk(Shape):

    def __init__(self):
        super().__init__()
        num_points = 50
        t = list(range(0, num_points))
        t.append(0)
        t = np.array(t) / num_points * 2*np.pi
        self.x = 0.5*np.cos(t)
        self.y = 0.5*np.sin(t)


class ParameterSet:

    def __init__(self, scaling, ps_range, ps2_range, min_intensity,
                 max_num_shape, scale, rot, img_size, max_intensity, g_blur):

        self.ps = scaling(np.random.rand(1)[0], ps_range)
        self.ps2 = scaling(np.random.rand(1)[0], ps2_range)
        self.min_intensity = scaling(np.random.rand(1)[0], min_intensity)
        self.blur_sigma = scaling(np.random.rand(1)[0], g_blur)

        self.max_num_shape = max_num_shape
        self.num_trig = np.random.randint(max_num_shape + 1)
        self.num_rect = np.random.randint(max_num_shape + 1)
        self.num_disk = np.random.randint(max_num_shape + 1)

        self.trigs = []
        for shape_params in range(self.num_trig):
            self.trigs.append(
                ShapeParams(scaling, scale, rot, img_size, max_intensity))

        self.rects = []
        for shape_params in range(self.num_rect):
            self.rects.append(
                ShapeParams(scaling, scale, rot, img_size, max_intensity))

        self.disks = []
        for shape_params in range(self.num_disk):
            self.disks.append(
                ShapeParams(scaling, scale, rot, img_size, max_intensity))

    def get_csv_output_line(self):

        params_list = [self.ps, self.ps2, self.min_intensity, self.blur_sigma,
                       self.max_num_shape,
                       self.num_trig, self.num_rect, self.num_disk]

        num_shape_params = 5
        k = 0
        for shape_params in self.trigs:
            params_list += [shape_params.scale, shape_params.rot,
                            shape_params.x, shape_params.y,
                            shape_params.intensity]
            k += 1
        if k < self.max_num_shape:
            params_list += ['nan'] * num_shape_params * (self.max_num_shape-k)

        k = 0
        for shape_params in self.rects:
            params_list += [shape_params.scale, shape_params.rot,
                            shape_params.x, shape_params.y,
                            shape_params.intensity]
            k += 1
        if k < self.max_num_shape:
            params_list += ['nan'] * num_shape_params * (self.max_num_shape-k)

        k = 0
        for shape_params in self.disks:
            params_list += [shape_params.scale, shape_params.rot,
                            shape_params.x, shape_params.y,
                            shape_params.intensity]
            k += 1
        if k < self.max_num_shape:
            params_list += ['nan'] * num_shape_params * (self.max_num_shape-k)

        return params_list

    def get_csv_header(self):

        header_list = ['particle_density', 'particle_s2_density',
                       'min_intensity', 'blur_sigma',
                       'max_num_shape',
                       'num_trig', 'num_rect', 'num_disk']

        header_trigs = ['trig_scale', 'trig_rot', 'trig_x', 'trig_y',
                        'trig_intensity'] * self.max_num_shape
        header_rects = ['rect_scale', 'rect_rot', 'rect_x', 'rect_y',
                        'rect_intensity'] * self.max_num_shape
        header_disks = ['disk_scale', 'disk_rot', 'disk_x', 'disk_y',
                        'disk_intensity'] * self.max_num_shape

        return header_list + header_trigs + header_rects + header_disks


class ShapeParams:

    def __init__(self, scaling, scale, rot, img_size, max_intensity):

        self.scale = scaling(np.random.rand(1)[0], scale)
        self.rot = scaling(np.random.rand(1)[0], rot)
        self.x = np.random.randint(img_size)
        self.y = np.random.randint(img_size)
        self.intensity = scaling(np.random.rand(1)[0], max_intensity)
