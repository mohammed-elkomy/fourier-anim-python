# Fourier Series Animation 
This repo holds the code for [my medium article](www.go.om) on animating Fourier series.

**To install the requirements**:
```
pip install -r requirements.txt
```
The directory structure for the repo: ⤵⤵
```
├── arrow_animation.py ➡ arrow animation generation script.
├── evolution_demo.py ➡ evolution animation generation script.
├── core ➡ core scripts and modules.
│ ├── fourier_drawer.py ➡ for drawing Fourier coefficients using opencv
│ ├── fourier_numerical_approximator.py ➡ finding coefficients.
│ └── generate_points.py ➡ generate the PTS files.
├── data ➡ SVG files + PTS files, example fourier.svg.
├── demos ➡ demo GIFs for repo preview.
└── example
│ ├── bezier.py ➡ making random smooth curves.
│ ├── generate_joseph_fourier_portrait.py ➡ generate the PTS file for fourier.svg
```

**How to use**:
1. To generate an arrow animation  
    ```
    python arrow_animation.py
    ```
    This will generate the arrow animation for the points fed to ```FOUR.animate(points,...)``` method, check the examples in the script
2. To generate an arrow animation  
    ```
    evolution_demo.py
    ```
    This will generate evolution animation for the points fed to ```FOUR.evolve(points,...)``` method, check the examples in the script

# Fun Zone 🤖
![Cake](https://github.com/mohammed-elkomy/fourier-anim-python/blob/main/demos/cake.gif)
![Eid](https://github.com/mohammed-elkomy/fourier-anim-python/blob/main/demos/eid.gif)
![fourier arrow](https://github.com/mohammed-elkomy/fourier-anim-python/blob/main/demos/fourier%20arrow.gif)
![fourier evolve](https://github.com/mohammed-elkomy/fourier-anim-python/blob/main/demos/fourier%20evolve.gif)
![heart](https://github.com/mohammed-elkomy/fourier-anim-python/blob/main/demos/heart.gif)
![mosque](https://github.com/mohammed-elkomy/fourier-anim-python/blob/main/demos/mosque.gif)
![thanks](https://github.com/mohammed-elkomy/fourier-anim-python/blob/main/demos/thanks.gif)

I don't own the rights for any of those images