from utils.data import *
import numpy as np
import matplotlib.pyplot as plt

N = 400


def main() -> int:
    class_0 = Data(mu=(2,  3), std=(.8,  2.5), n=N)
    class_1 = Data(mu=(5,  6), std=(1.2, 1.9), n=N)
    class_2 = Data(mu=(8,  1), std=(.9,   .9), n=N)
    class_3 = Data(mu=(15, 4), std=(.5,  2.0), n=N)
    
    x0, y0 = class_0.sample_initialize()
    x1, y1 = class_1.sample_initialize()
    x2, y2 = class_2.sample_initialize()
    x3, y3 = class_3.sample_initialize()

    plt.plot(x0, y0, "o", label="Classe 0")
    plt.plot(x1, y1, "o", label="Classe 1")
    plt.plot(x2, y2, "o", label="Classe 2")
    plt.plot(x3, y3, "o", label="Classe 3")

    plt.legend()

    plt.title("Plot das classes")

    plt.show()

    return 0

if __name__ == "__main__":
    main()