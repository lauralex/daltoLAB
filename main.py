import math
from pathlib import Path

import colour
import imageio.v3 as iio
import numpy as np
from PIL import Image
from colormath.color_conversions import convert_color
from colormath.color_objects import sRGBColor, LabColor

factor_g = 0.8
factor_r = 0.6


def applied_func(input):
    # converted: LabColor = convert_color(sRGBColor(*input, is_upscaled=True), LabColor)
    converted: LAB = LAB.from_rgb(*input, 0)
    # converted_1 = LAB.from_rgb(*input, 0)
    algorithm = 1
    if algorithm == 1:
        adjusted_aval = converted.lab_a / 128.0
        adjusted_lval = converted.lab_l
        if converted.lab_a >= 0 and converted.lab_l < 85:
            if converted.lab_l < 65:
                adjusted_lval = converted.lab_l * 1.2
            adjusted_aval = ((1 - converted.lab_a / 128.0) * factor_r + converted.lab_a / 128.0)
        elif converted.lab_l < 85:
            if converted.lab_l > 65:
                adjusted_lval = converted.lab_l * 0.8
            adjusted_aval = converted.lab_a * -1
            adjusted_aval = ((1 - adjusted_aval / 128.0) * factor_g + adjusted_aval / 128.0)
            adjusted_aval *= -1
        adjusted_bval = converted.lab_b / 128.0
        if converted.lab_b >= 0 and converted.lab_a >= 0 and converted.lab_l < 85:
            adjusted_bval = ((1 - converted.lab_b / 128.0) * factor_r - converted.lab_b / 128.0)
        # elif converted.lab_l < 85:
        #     adjusted_bval = converted.lab_b * -1
        #     adjusted_bval = ((1 - adjusted_bval / 128.0) * factor_g + adjusted_bval / 128.0)
        #     adjusted_bval *= -1
    else:
        adjusted_aval = converted.lab_a * factor_r

    # rgb_color: sRGBColor = convert_color(
    #     LabColor(converted.lab_l, adjusted_aval * 128.0 if algorithm == 1 else adjusted_aval,
    #              adjusted_bval * 128.0 if algorithm == 1 else converted.lab_b, illuminant='d65', observer='2'),
    #     sRGBColor)
    # rgb_compound = LAB(converted.lab_l, adjusted_aval * 128.0, adjusted_bval).rgb()
    # r, g, b = (0xFF & (rgb_compound >> 16), 0xFF & (rgb_compound >> 8), 0xFF & rgb_compound)

    rgb_from_lab = LAB(adjusted_lval if algorithm == 1 else converted.lab_l, adjusted_aval * 128.0 if algorithm == 1 else adjusted_aval,
                       adjusted_bval * 128.0 if algorithm == 1 else converted.lab_b).rgb()
    r, g, b = (0xFF & (rgb_from_lab >> 16), 0xFF & (rgb_from_lab >> 8), 0xFF & rgb_from_lab)
    return np.array([r, g, b], dtype=np.uint8)


def convert_image(image):
    return np.apply_along_axis(applied_func, 2, image)


def main():
    a = LAB.from_rgb(23, 200, 23, 0)

    image_path = Path('mammamia.png')
    my_image = iio.imread(image_path)

    is_alpha_present = my_image.shape[2] > 3
    if is_alpha_present:
        my_image = np.delete(my_image, 3, 2)

    my_image_converted: np.ndarray = convert_image(my_image)

    iio.imwrite(f'{image_path.stem}_converted{image_path.suffix}', my_image_converted)
    hex_color = a.hex()
    print(hex_color)


class LAB:
    def __init__(self, l, a, b, c=-1, s=-1):
        self.lab_l = l
        self.lab_a = a
        self.lab_b = b
        self.w = []
        self.c = c
        self.s = s

    def hex(self):
        rgb = self.rgb()
        r = (0xFF & (rgb >> 16))
        g = (0xFF & (rgb >> 8))
        b = (0xFF & rgb)
        sr = hex(r)[2:]
        sg = hex(g)[2:]
        sb = hex(b)[2:]
        if len(sr) < 2:
            sr = '0' + sr
        if len(sg) < 2:
            sg = '0' + sg
        if len(sb) < 2:
            sb = "0" + sb
        return "#" + sr + sg + sb

    def __hash__(self) -> int:
        x = int(self.lab_l)
        y = int(self.lab_a + 110)
        z = int(self.lab_b + 110)
        return (x << 16) | (y << 8) | z

    def distance(self, y):
        dl = self.lab_l - y.lab_l
        da = self.lab_a - y.lab_a
        db = self.lab_b - y.lab_b
        return math.sqrt(dl * dl + da * da + db * db)

    def rgb(self):
        # first, map CIE L*a*b* to CIE XYZ
        y = (self.lab_l + 16) / 116
        x = y + self.lab_a / 500
        z = y - self.lab_b / 200

        # D65 standard referent
        x_const = 0.950470
        y_const = 1.0
        z_const = 1.088830
        x = x_const * (x * x * x if x > 0.206893034 else (x - 4.0 / 29) / 7.787037)
        y = y_const * (y * y * y if y > 0.206893034 else (y - 4.0 / 29) / 7.787037)
        z = z_const * (z * z * z if z > 0.206893034 else (z - 4.0 / 29) / 7.787037)

        # second, map CIE XYZ to sRGB
        r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
        g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
        b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
        r = 12.92 * r if r <= 0.00304 else 1.055 * math.pow(r, 1 / 2.4) - 0.055
        g = 12.92 * g if g <= 0.00304 else 1.055 * math.pow(g, 1 / 2.4) - 0.055
        b = 12.92 * b if b <= 0.00304 else 1.055 * math.pow(b, 1 / 2.4) - 0.055

        # third, get sRGB values
        ir = int(round(255 * r))
        ir = max(0, min(ir, 255))
        ig = int(round(255 * g))
        ig = max(0, min(ig, 255))
        ib = int(round(255 * b))
        ib = max(0, min(ib, 255))
        return (0xFF0000 & (ir << 16)) | (0x00FF00 & (ig << 8)) | (0xFF & ib)

    @staticmethod
    def from_rgb(ri, gi, bi, bin_size):
        r = ri / 255.0
        g = gi / 255.0
        b = bi / 255.0

        # D65 standard referent double
        x_const = 0.950470
        y_const = 1.0
        z_const = 1.088830

        # second, map sRGB to CIE XYZ
        r = r / 12.92 if r <= 0.04045 else math.pow((r + 0.055) / 1.055, 2.4)
        g = g / 12.92 if g <= 0.04045 else math.pow((g + 0.055) / 1.055, 2.4)
        b = b / 12.92 if b <= 0.04045 else math.pow((b + 0.055) / 1.055, 2.4)

        x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / x_const
        y = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b) / y_const
        z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / z_const

        # third, map CIE XYZ to CIE l * a * b * and return
        x = math.pow(x, 1.0 / 3) if x > 0.008856 else 7.787037 * x + 4.0 / 29
        y = math.pow(y, 1.0 / 3) if y > 0.008856 else 7.787037 * y + 4.0 / 29
        z = math.pow(z, 1.0 / 3) if z > 0.008856 else 7.787037 * z + 4.0 / 29

        l = 116 * y - 16
        a = 500 * (x - y)
        b = 200 * (y - z)

        if bin_size > 0:
            l = bin_size * math.floor(l / bin_size)
            a = bin_size * math.floor(a / bin_size)
            b = bin_size * math.floor(b / bin_size)

        return LAB(l, a, b)

    @staticmethod
    def from_rgb_r(ri, gi, bi, bin_size):
        # first, normalize RGB values double
        r = ri / 255.0
        g = gi / 255.0
        b = bi / 255.0

        # D65 standard referent double
        x_const = 0.950470
        y_const = 1.0
        z_const = 1.088830

        # second, map sRGB to CIE XYZ
        r = r / 12.92 if r <= 0.04045 else math.pow((r + 0.055) / 1.055, 2.4)
        g = g / 12.92 if g <= 0.04045 else math.pow((g + 0.055) / 1.055, 2.4)
        b = b / 12.92 if b <= 0.04045 else math.pow((b + 0.055) / 1.055, 2.4)

        x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / x_const
        y = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b) / y_const
        z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / z_const

        # third, map CIE XYZ to CIE l * a * b * and return
        x = math.pow(x, 1.0 / 3) if x > 0.008856 else 7.787037 * x + 4.0 / 29
        y = math.pow(y, 1.0 / 3) if y > 0.008856 else 7.787037 * y + 4.0 / 29
        z = math.pow(z, 1.0 / 3) if z > 0.008856 else 7.787037 * z + 4.0 / 29

        l = 116 * y - 16
        a = 500 * (x - y)
        b = 200 * (y - z)

        if bin_size > 0:
            l = bin_size * round(l / bin_size)
            a = bin_size * round(a / bin_size)
            b = bin_size * round(b / bin_size)

        return LAB(l, a, b)

    @staticmethod
    def is_in_rgb_gamut(l, a, b):
        # first, map CIE L*a*b* to CIE XYZ
        y = (l + 16) / 116
        x = y + a / 500
        z = y - b / 200
        # D65 standard referent
        x_const = 0.950470
        y_const = 1.0
        z_const = 1.088830
        x = x_const * (x * x * x if x > 0.206893034 else (x - 4.0 / 29) / 7.787037)
        y = y_const * (y * y * y if y > 0.206893034 else (y - 4.0 / 29) / 7.787037)
        z = z_const * (z * z * z if z > 0.206893034 else (z - 4.0 / 29) / 7.787037)
        # second, map CIE XYZ to sRGB
        r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
        g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
        b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z

        r = 12.92 * r if r <= 0.00304 else 1.055 * math.pow(r, 1 / 2.4) - 0.055
        g = 12.92 * g if g <= 0.00304 else 1.055 * math.pow(g, 1 / 2.4) - 0.055
        b = 12.92 * b if b <= 0.00304 else 1.055 * math.pow(b, 1 / 2.4) - 0.055

        # third, check sRGB values
        return not (r < 0 or r > 1 or g < 0 or g > 1 or b < 0 or b > 1)

    @staticmethod
    def ciede2000(x, y):
        # adapted from Sharma et al's MATLAB implementation at
        # http://www.ece.rochester.edu/~gsharma/ciede2000/

        # parametric factors, use defaults
        kl = 1
        kc = 1
        kh = 1

        # compute terms
        pi = math.pi
        l1 = x.lab_l
        a1 = x.lab_a
        b1 = x.lab_b
        cab1 = math.sqrt(a1 * a1 + b1 * b1)
        l2 = y.lab_l
        a2 = y.lab_a
        b2 = y.lab_b
        cab2 = math.sqrt(a2 * a2 + b2 * b2)
        cab = 0.5 * (cab1 + cab2)
        g = 0.5 * (1 - math.sqrt(math.pow(cab, 7) / (math.pow(cab, 7) + math.pow(25, 7))))
        ap1 = (1 + g) * a1
        ap2 = (1 + g) * a2
        cp1 = math.sqrt(ap1 * ap1 + b1 * b1)
        cp2 = math.sqrt(ap2 * ap2 + b2 * b2)
        cpp = cp1 * cp2

        # ensure hue is between 0 and 2pi
        hp1 = math.atan2(b1, ap1)
        if hp1 < 0:
            hp1 += 2 * pi
        hp2 = math.atan2(b2, ap2)
        if hp2 < 0:
            hp2 += 2 * pi

        d_l = l2 - l1
        d_c = cp2 - cp1
        dhp = hp2 - hp1

        if dhp > +pi:
            dhp -= 2 * pi
        if dhp < -pi:
            dhp += 2 * pi
        if cpp == 0:
            dhp = 0

        # Note that the defining equations actually need
        # signed Hue and chroma differences which is different
        # from prior color difference formulae
        d_h = 2 * math.sqrt(cpp) * math.sin(dhp / 2)

        # Weighting functions
        lp = 0.5 * (l1 + l2)
        cp = 0.5 * (cp1 + cp2)

        # Average Hue Computation
        # This is equivalent to that in the paper but simpler programmatically.
        # Average hue is computed in radians and converted to degrees where needed
        hp = 0.5 * (hp1 + hp2)
        # Identify positions for which abs hue diff exceeds 180 degrees
        if abs(hp1 - hp2) > pi:
            hp -= pi
        if hp < 0:
            hp += 2 * pi

        # Check if one of the chroma values is zero, in which case set
        # mean hue to the sum which is equivalent to other value
        if cpp == 0:
            hp = hp1 + hp2

        lpm502 = (lp - 50) * (lp - 50)
        sl = 1 + 0.015 * lpm502 / math.sqrt(20 + lpm502)
        sc = 1 + 0.045 * cp
        t = 1 - 0.17 * math.cos(hp - pi / 6) + 0.24 * math.cos(2 * hp) + 0.32 * math.cos(
            3 * hp + pi / 30) - 0.20 * math.cos(4 * hp - 63 * pi / 180)
        sh = 1 + 0.015 * cp * t
        ex = (180 / pi * hp - 275) / 25
        delthetarad = (30 * pi / 180) * math.exp(-1 * (ex * ex))
        rc = 2 * math.sqrt(math.pow(cp, 7) / (math.pow(cp, 7) + math.pow(25, 7)))
        rt = -1 * math.sin(2 * delthetarad) * rc

        d_l = d_l / (kl * sl)
        d_c = d_c / (kc * sc)
        d_h = d_h / (kh * sh)

        # The CIE 00 color difference
        return math.sqrt(d_l * d_l + d_c * d_c + d_h * d_h + rt * d_c * d_h)


if __name__ == '__main__':
    main()
