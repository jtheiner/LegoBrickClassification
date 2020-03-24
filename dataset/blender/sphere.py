import numpy as np



def get_positions(theta_range=(0, np.pi * 2),
                  phi_range=(0, np.pi / 2),
                  radius=1,
                  step_size=1000,
                  n_points_circle=1000,
                  zlow=0.0,
                  zhigh=1.0):
    theta = np.linspace(theta_range[0], theta_range[1], step_size)
    phi = np.linspace(phi_range[0], phi_range[1], step_size)

    thetam, phim = np.meshgrid(theta, phi)
    thetam = np.ravel(thetam)
    phim = np.ravel(phim)
    # calculate coordinates
    x = radius * np.cos(thetam) * np.sin(phim)
    y = radius * np.sin(thetam) * np.sin(phim)
    z = radius * np.cos(phim)
    # sort by z value
    idx = np.argsort(z)
    x = x[idx]
    y = y[idx]
    z = z[idx]

    # keep a uniform distribution
    p = 1 - (z / np.max(z))
    p = p / np.sum(p)  # normalize to sum 1
    idx_random = np.random.choice(np.arange(len(z)), n_points_circle, replace=False, p=p)
    x = np.squeeze(x[idx_random])
    y = np.squeeze(y[idx_random])
    z = np.squeeze(z[idx_random])

    # clip upper
    idx_zlimit_upper = np.argwhere(z <= radius * zhigh)
    x = np.squeeze(x[idx_zlimit_upper])
    y = np.squeeze(y[idx_zlimit_upper])
    z = np.squeeze(z[idx_zlimit_upper])

    # clip lower
    idx_zlimit_lower = np.argwhere(z >= radius * zlow)
    x = np.squeeze(x[idx_zlimit_lower])
    y = np.squeeze(y[idx_zlimit_lower])
    z = np.squeeze(z[idx_zlimit_lower])

    return x, y, z, list(zip(x, y, z))


if __name__ == '__main__':

    x, y, z, _ = get_positions()
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x, y, z, s=1)
    ax.set_xlim3d(-1., 1.)
    ax.set_ylim3d(-1., 1.)
    ax.set_zlim3d(0., 1.)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


    heatmap, xedges, yedges = np.histogram2d(x, y, bins=33)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.colorbar()
    plt.show()

    plt.clf()
    histz = np.histogram(z, bins=10)
    plt.plot(histz[1])
    plt.show()
