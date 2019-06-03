import numpy as np
import sympy as sp
from itertools import permutations
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# title size
FONT_SIZE_FIG_TITLE = 16
FONT_SIZE_SP_TITLE = 16
FONT_SIZE_AXIS_LABEL = 14
FONT_SIZE_LEGEND = 14


def beta_pdf(x, a, b):
    """
    :param x: Beta random variable--a scalar SymPy symbol
    :param a: Beta parameter--a scalar SymPy symbol
    :param b: Beta parameter--a scalar SymPy symbol
    :return: a SymPy expression for the Beta pdf
    """
    # split x into it's fractional components (required for the simplification engine to work)
    x_numer, x_denom = sp.fraction(x)

    # define the distributions
    pdf = sp.gamma(a + b) / sp.gamma(a) / sp.gamma(b) *\
          x_numer ** (a - 1) * x_denom ** (1 - a) * \
          x_denom ** (1 - b) * (x_denom - x_numer) ** (b - 1)

    return pdf


def kumaraswamy_pdf(x, a, b):
    """
    :param x: Kumaraswamy random variable--a scalar SymPy symbol
    :param a: Kumaraswamy parameter--a scalar SymPy symbol
    :param b: Kumaraswamy parameter--a scalar SymPy symbol
    :return: a SymPy expression for the Kumaraswamy pdf
    """
    # define the distributions
    pdf = a * b * x ** (a-1) * (1 - x ** a) ** (b - 1)

    return pdf


def stick_breaking(K, pdf):
    """
    :param K: number of dimensions
    :param pdf: a function--either beta_pdf or kumaraswamy_pdf
    :return:    expected_f--the expected pdf w.r.t. all sampling orders
                f--a list of pdf for each sampling order
                x--a tuple of symbols
                a--a tuple of symbols
    """
    # make symbol names
    X = sp.symbols(' '.join(['X_{:d}'.format(i + 1) for i in range(K)]), real=True, nonnegative=True)
    A = sp.symbols(' '.join(['A_{:d}'.format(i + 1) for i in range(K)]), real=True, postive=True)

    # generate all permutations
    perms = list(permutations(range(K)))

    # loop over the permutations
    f = []
    for p in perms:

        # initialize the joint pdf for this permutation
        f_joint = pdf(X[p[0]], A[p[0]], sum([A[j] for j in set(p) - set(p[:1])]))

        # loop over the free dimensions
        for i in range(1, len(p) - 1):

            # set the stick-breaking parameters
            a = A[p[i]]
            b = [A[j] for j in set(p) - set(p[:i + 1])]

            # compute the amount of stick remaining
            x_left = (1 - sum([X[j] for j in p[:i]])) ** (-1)

            # multiply the current joint distribution by the next dimension's conditional distribution
            f_joint = sp.powsimp(f_joint * x_left * pdf(X[p[i]] * x_left, a, sum(b)))

        # append the permutation
        f.append(f_joint)

    # take the expectation
    expected_f = sum(f) / len(f)

    return expected_f, f, X, A, perms


class Dirichlet(object):
    def __init__(self, a):
        """
        :param a: K-dimensional parameter vector
        """
        # construct the pdf using the expected stick breaking process and save associated symbols
        self.expected_f, self.f, self.x, self.a, self.perms = stick_breaking(K=len(a), pdf=beta_pdf)

        # substitute variables for the non-free dimension
        for x in self.x:
            subs = 1 - sum(list(set(self.x) - {x}))
            self.expected_f = self.expected_f.subs(subs, x)
            self.f = list(map(lambda f: f.subs(subs, x), self.f))

        # print resulting expression
        print('Dirichlet:', 'E[f(x;a)] =', self.expected_f)

        # substitute in parameter values
        self.expected_f = self.expected_f.subs(dict(zip(self.a, a)))
        self.f = [f.subs(dict(zip(self.a, a))) for f in self.f]

    def pdf(self, x, order=-1):
        """
        :param x: (K-1)-dimensional value for the degrees of freedom
        :param order: which sampling order pdf to use, -1 uses the expected pdf
        :return: f(x;a)
        """
        assert order == -1 or order in range(len(self.f))

        # evaluate the pdf
        if order < 0:
            return np.float64(self.expected_f.evalf(subs=dict(zip(self.x, x)), chop=True))
        else:
            return np.float64(self.f[order].evalf(subs=dict(zip(self.x, x)), chop=True))


class MultivariateKumaraswamy(object):
    def __init__(self, a):
        """
        :param a: K-dimensional parameter vector
        """
        # construct the pdf using the expected stick breaking process and save associated symbols
        self.expected_f, self.f, self.x, self.a, self.perms = stick_breaking(K=len(a), pdf=kumaraswamy_pdf)

        # print resulting expression
        print('MV Kumaraswamy:', 'E[f(x;a)] =', self.expected_f)

        # substitute in parameter values
        self.expected_f = self.expected_f.subs(dict(zip(self.a, a)))
        self.f = [f.subs(dict(zip(self.a, a))) for f in self.f]

    def pdf(self, x, order=-1):
        """
        :param x: (K-1)-dimensional value for the degrees of freedom
        :param order: which sampling order pdf to use, -1 uses the expected pdf
        :return: f(x;a)
        """
        assert order == -1 or order in range(len(self.f))

        # evaluate the pdf
        if order < 0:
            return np.float64(self.expected_f.evalf(subs=dict(zip(self.x, x)), chop=True))
        else:
            return np.float64(self.f[order].evalf(subs=dict(zip(self.x, x)), chop=True))


def get_asymmetry(dist, order, pi, symmetry):
    """
    :param dist: the target distribution class object
    :param order: the specified stick-breaking used to generate the pdf
    :param pi: an NumPy array of dimensions (number of evaluation points, K)
    :param symmetry: an iterable of length K-1 that defines the axis of symmetry
    :return: a NumPy array of shape(pi) that captures any anti-symmetry
    """
    # initialize the anti-symmetry measurements
    asymmetry = np.zeros(pi.shape[0])
    captured = np.zeros(pi.shape[0])

    # loop over the points
    for i in range(len(pi)):

        # find point of symmetry
        i_sym = np.argmin(sum([np.abs(pi[i, p[0]] - pi[:, p[1]]) for p in permutations(symmetry)]))

        # save the measurement and mark its capture
        asymmetry[i] = np.abs(dist.pdf(pi[i], order=order) - dist.pdf(pi[i_sym], order=order))
        captured[i] = 1

    # make sure we caught them all
    assert np.sum(captured) == len(pi)

    # clamp differences to numerical relevance
    asymmetry[asymmetry < 1e-9] = 0

    return asymmetry


def plot_asymmetries_2_dimensions(nlevels=200):

    # define some K=2 parameters
    a = [np.array([1 / 2, 1 / 2]), np.array([1 / 2, 2]), np.array([2, 1 / 2]), np.array([2, 2])]

    # convert to Barycentric coordinates
    x1 = np.expand_dims(np.linspace(0.05, 0.95, nlevels), axis=-1)
    pis = np.concatenate((x1, 1 - x1), axis=-1)

    # construct the figure
    fig, ax = plt.subplots(1, 4, figsize=(16, 5))
    ax = np.reshape(ax, -1)
    line_width = 1.5

    # loop over the parameters
    for i in range(len(a)):

        # construct the distributions
        dist_diric = Dirichlet(a[i])
        dist_kumar = MultivariateKumaraswamy(a[i])

        # plot Kumaraswamy distributions
        ax[i].plot(x1, [dist_kumar.pdf(pi, order=-1) for pi in pis],
                   linewidth=line_width,
                   label='E[$f(x;\\alpha)$]')
        ax[i].plot(x1, [dist_kumar.pdf(pi, order=0) for pi in pis],
                   linestyle='--',
                   linewidth=line_width,
                   label='$f_{12}(x;\\alpha)$')
        ax[i].plot(x1, [dist_kumar.pdf(pi, order=1) for pi in pis],
                   linestyle='--',
                   linewidth=line_width,
                   label='$f_{21}(x;\\alpha)$')

        # plot Dirichlet distribution
        ax[i].plot(x1, [dist_diric.pdf(pi, order=-1) for pi in pis],
                   color='k',
                   alpha=0.5,
                   linestyle=':',
                   linewidth=line_width * 2,
                   label='Dirichlet($x;\\alpha$)')

        # set limits
        ax[i].set_xlim([0, 1])

        # add title
        ax[i].set_title('$\\alpha_1 = {:.1f}, \\alpha_2 = {:.1f}$'.format(a[i][0], a[i][1]), fontsize=FONT_SIZE_SP_TITLE)

    # make it tight
    plt.subplots_adjust(left=0.02, bottom=0.18, right=0.99, top=0.93, wspace=0.125, hspace=0.35)

    # add legend
    ax[-1].legend(ncol=4, bbox_to_anchor=(-0.15, -0.075), fontsize=FONT_SIZE_LEGEND)


def plot_asymmetries_3_dimensions(dist, title=None, nlevels=200, subdiv=4):

    # define the simplex (an equilateral triangle with vertices (0,1), (1,0), and (0.5, 0.5 * tan(60 degrees)
    corners = np.array([[0, 0], [1, 0], [0.5, 0.5 * np.tan(np.pi / 3)]])
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])

    # compute the cartesian coordinates for the midpoint of each side
    midpoints = [(corners[(i + 1) % 3] + corners[(i + 2) % 3]) / 2.0 for i in range(3)]

    def cartesian_to_barycentric(xy, tol=1.e-3):

        # convert Cartesian coordinates to Barycentric
        s = np.array([(corners[i] - midpoints[i]).dot(xy - midpoints[i]) / 0.75 for i in range(3)])

        # ensure we are not to close to 0 or 1 for numerical reasons
        s[s < tol] = tol
        s[s > 1 - tol] = 1 - tol

        # normalize the clipped result
        s /= np.sum(s)

        return s

    # set evaluation points
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)

    # convert to Barycentric coordinates
    pi = np.array([cartesian_to_barycentric(xy) for xy in zip(trimesh.x, trimesh.y)])

    # construct the figure
    rows = 4
    cols = 7
    fig, ax = plt.subplots(rows, cols, figsize=(8, 5))
    ax = np.reshape(ax, -1)
    i_plot = 0

    # plot the expected (symmetric under uniform permutations) distribution
    e_f = [dist.pdf(pi, order=-1) for pi in pi]
    ax[i_plot].tricontourf(trimesh, e_f, nlevels)
    ax[i_plot].set_title('$E[f]$', fontsize=FONT_SIZE_SP_TITLE)
    ax[i_plot].set_ylabel('PDF', fontsize=FONT_SIZE_AXIS_LABEL)
    i_plot += 1

    # loop over the pdf's associated with each stick-breaking order
    f = []
    for i in range(len(dist.f)):

        # evaluate the probabilities
        f.append([dist.pdf(pi, order=i) for pi in pi])

        # get the ordering
        order = ''.join([str(o + 1) for o in dist.perms[i]])

        # plot the data
        ax[i_plot].tricontourf(trimesh, f[-1], nlevels)
        ax[i_plot].set_title('$f_{' + order + '}$', fontsize=FONT_SIZE_SP_TITLE)
        i_plot += 1

    # define possible symmetries
    axes = {0, 1, 2}
    symmetries = [[1, 2], [0, 2], [0, 1]]

    # loop over the symmetries
    for symmetry in symmetries:

        # loop over the orders
        for order in range(-1, len(f)):

            # plot anti-symmetric portion
            ax[i_plot].tricontourf(trimesh, get_asymmetry(dist, order, pi, symmetry), nlevels)
            if np.mod(i_plot, cols) == 0:
                axis = list(axes - set(symmetry))[0]
                ax[i_plot].set_ylabel('$x_{' + str(axis + 1) + '}$ Asym.', fontsize=FONT_SIZE_AXIS_LABEL)
            i_plot += 1

    # make it pretty
    for i in range(len(ax)):
        ax[i].axis('equal')
        ax[i].set_xlim(0, 1)
        ax[i].set_ylim(0, 0.75**0.5)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].spines['left'].set_visible(False)

    # add the title if one is provided
    if title is not None:
        fig.suptitle(title, fontsize=FONT_SIZE_FIG_TITLE)

    # make it tight
    plt.subplots_adjust(left=0.04, bottom=0, right=1, top=0.85, wspace=0, hspace=0)


# plot asymmetries
plot_asymmetries_2_dimensions()

# hold the plot
plt.show()

# define some K=3 parameters
a = [np.array([1, 3, 3]),
     np.array([1/3, 1/3, 1/3]),
     np.array([1, 1, 1]),
     np.array([3, 3, 3])]

# plot the results for theses parameters
for i in range(len(a)):

    # plot multivariate Kumaraswamy
    title = 'MV Kumaraswamy with $\\alpha_1 = {:.2f}, \\alpha_2 = {:.2f}, \\alpha_3 = {:.2f}$'.format(a[i][0],
                                                                                                      a[i][1],
                                                                                                      a[i][2])
    plot_asymmetries_3_dimensions(MultivariateKumaraswamy(a[i]), title=title)

    # plot Dirichlet
    title = 'Dirichlet with $\\alpha_1 = {:.2f}, \\alpha_2 = {:.2f}, \\alpha_3 = {:.2f}$'.format(a[i][0],
                                                                                                 a[i][1],
                                                                                                 a[i][2])
    plot_asymmetries_3_dimensions(Dirichlet(a[i]), title=title)

    # hold the plot
    plt.show()
