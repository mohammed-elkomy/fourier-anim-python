"""
Example script to generate joseph fourier portrait
1) an  input svg file is given and paths are represented in intermediate format
2) this intermediate format is fed to our drawing scripts "arrow_animation.py" or "evolution_demo.py"
"""

import cv2

from core.generate_points import generate_points_from_svg, animate_points

generate_points_from_svg("../data/fourier.svg")
animate_points("../data/fourier.pts")
print("Press any key to continue")
cv2.waitKey()
